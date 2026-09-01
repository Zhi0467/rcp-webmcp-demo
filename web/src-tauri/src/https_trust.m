// App-scoped TLS trust for RCP's desktop-owned local HTTPS origins.
//
// WKWebView trusts exactly one certificate that the app generated for itself,
// without installing anything into a system-wide trust store. wry's
// `WryNavigationDelegate` does not implement the server-trust challenge, so
// this file adds that one method to the existing class at runtime and pins a
// single DER SHA-256. Every other certificate is refused.
//
// The same pinning primitive is used by the retained local-HTTPS probe.

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <WebKit/WebKit.h>
#import <objc/runtime.h>

typedef void (^RcpChallengeCompletion)(NSURLSessionAuthChallengeDisposition disposition,
                                       NSURLCredential *_Nullable credential);
typedef void (*RcpCookieCompletion)(void *context, int code);

static NSString *gPinnedFingerprint = nil;
static int gAccepted = 0;
static int gRefused = 0;
static int gDelegated = 0;

static NSString *RcpLeafFingerprint(SecTrustRef trust) {
  if (trust == NULL) {
    return nil;
  }
  CFArrayRef chain = SecTrustCopyCertificateChain(trust);
  if (chain == NULL) {
    return nil;
  }
  NSString *result = nil;
  if (CFArrayGetCount(chain) > 0) {
    SecCertificateRef leaf = (SecCertificateRef)CFArrayGetValueAtIndex(chain, 0);
    CFDataRef der = SecCertificateCopyData(leaf);
    if (der != NULL) {
      unsigned char digest[CC_SHA256_DIGEST_LENGTH];
      CC_SHA256(CFDataGetBytePtr(der), (CC_LONG)CFDataGetLength(der), digest);
      NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
      for (int index = 0; index < CC_SHA256_DIGEST_LENGTH; index += 1) {
        [hex appendFormat:@"%02x", digest[index]];
      }
      result = hex;
      CFRelease(der);
    }
  }
  CFRelease(chain);
  return result;
}

static void RcpHandleChallenge(id self, SEL _cmd, id webView,
                               NSURLAuthenticationChallenge *challenge,
                               RcpChallengeCompletion completionHandler) {
  (void)self;
  (void)_cmd;
  (void)webView;
  NSURLProtectionSpace *space = challenge.protectionSpace;
  if (![space.authenticationMethod isEqualToString:NSURLAuthenticationMethodServerTrust]) {
    gDelegated += 1;
    completionHandler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
    return;
  }

  SecTrustRef trust = space.serverTrust;
  NSString *observed = RcpLeafFingerprint(trust);
  if (observed != nil && gPinnedFingerprint != nil &&
      [observed caseInsensitiveCompare:gPinnedFingerprint] == NSOrderedSame) {
    gAccepted += 1;
    fprintf(stderr, "[https-trust] accepted pinned certificate for %s\n",
            space.host.UTF8String ?: "<unknown>");
    completionHandler(NSURLSessionAuthChallengeUseCredential,
                      [NSURLCredential credentialForTrust:trust]);
    return;
  }

  gRefused += 1;
  fprintf(stderr, "[https-trust] refused unpinned certificate for %s (sha256=%s)\n",
          space.host.UTF8String ?: "<unknown>", observed.UTF8String ?: "<none>");
  completionHandler(NSURLSessionAuthChallengeCancelAuthenticationChallenge, nil);
}

static void RcpReportFailure(id self, SEL _cmd, id webView, id navigation, NSError *error) {
  (void)self;
  (void)webView;
  (void)navigation;
  fprintf(stderr, "[https-trust] %s domain=%s code=%ld reason=%s\n", sel_getName(_cmd),
          error.domain.UTF8String ?: "<none>", (long)error.code,
          error.localizedDescription.UTF8String ?: "<none>");
}

// Returns 0 on success, 1 when the delegate is absent, 2 when another
// implementation owns the selector, 3 when the runtime refused the method,
// and 4 on bad arguments. `webview` is the WKWebView pointer from Tauri's
// PlatformWebview: asking the live object for its delegate avoids guessing
// wry's class name.
int rcp_https_trust_install_pin(const char *fingerprint_hex, void *webview) {
  if (fingerprint_hex == NULL || webview == NULL) {
    return 4;
  }
  gPinnedFingerprint = [NSString stringWithUTF8String:fingerprint_hex];

  WKWebView *view = (__bridge WKWebView *)webview;
  id delegate_object = view.navigationDelegate;
  if (delegate_object == nil) {
    fprintf(stderr, "[https-trust] the webview has no navigation delegate\n");
    return 1;
  }
  Class delegate = object_getClass(delegate_object);
  fprintf(stderr, "[https-trust] navigation delegate class is %s\n", class_getName(delegate));
  SEL selector = @selector(webView:didReceiveAuthenticationChallenge:completionHandler:);
  Method existing = class_getInstanceMethod(delegate, selector);
  if (existing != NULL) {
    if (method_getImplementation(existing) != (IMP)RcpHandleChallenge) {
      return 2;
    }
  } else {
    if (!class_addMethod(delegate, selector, (IMP)RcpHandleChallenge, "v@:@@@?")) {
      return 3;
    }
  }
  if (![delegate_object respondsToSelector:selector]) {
    return 5;
  }
  // WKWebView caches which delegate methods exist when the delegate is
  // assigned. Reassigning the same object forces that table to be rebuilt so
  // the newly added challenge method is actually consulted.
  // Failure reporting is diagnostic only; wry implements neither selector.
  SEL provisional = @selector(webView:didFailProvisionalNavigation:withError:);
  if (class_getInstanceMethod(delegate, provisional) == NULL) {
    class_addMethod(delegate, provisional, (IMP)RcpReportFailure, "v@:@@@");
  }
  SEL committed = @selector(webView:didFailNavigation:withError:);
  if (class_getInstanceMethod(delegate, committed) == NULL) {
    class_addMethod(delegate, committed, (IMP)RcpReportFailure, "v@:@@@");
  }

  view.navigationDelegate = nil;
  view.navigationDelegate = delegate_object;
  fprintf(stderr, "[https-trust] challenge handler added and delegate reattached\n");

  // Cookie persistence across restart depends on this being a persistent
  // store, which an unbundled binary may not get.
  fprintf(stderr, "[https-trust] website data store persistent=%s\n",
          view.configuration.websiteDataStore.isPersistent ? "yes" : "no");

  return 0;
}

// Parse one server-issued __Host- session cookie with Foundation, validate the
// browser security attributes again, then place it directly in this WebView's
// persistent cookie store. The completion callback is invoked exactly once
// after WKWebView confirms the write. No raw session value is returned.
int rcp_https_trust_set_team_cookie(void *webview, const char *origin,
                                    const char *set_cookie,
                                    RcpCookieCompletion completion,
                                    void *context) {
  if (webview == NULL || origin == NULL || set_cookie == NULL || completion == NULL ||
      context == NULL) {
    return 4;
  }
  WKWebView *view = (__bridge WKWebView *)webview;
  NSURL *url = [NSURL URLWithString:[NSString stringWithUTF8String:origin]];
  NSString *header = [NSString stringWithUTF8String:set_cookie];
  if (url == nil || header == nil || ![url.scheme isEqualToString:@"https"] ||
      url.host.length == 0) {
    return 6;
  }
  NSArray<NSHTTPCookie *> *cookies =
      [NSHTTPCookie cookiesWithResponseHeaderFields:@{ @"Set-Cookie" : header }
                                             forURL:url];
  if (cookies.count != 1) {
    return 7;
  }
  NSHTTPCookie *cookie = cookies.firstObject;
  NSString *domain = cookie.domain;
  if ([domain hasPrefix:@"."]) {
    domain = [domain substringFromIndex:1];
  }
  if (![cookie.name isEqualToString:@"__Host-rcp_session"] || !cookie.isSecure ||
      !cookie.isHTTPOnly || ![cookie.path isEqualToString:@"/"] ||
      [domain caseInsensitiveCompare:url.host] != NSOrderedSame) {
    return 8;
  }
  [view.configuration.websiteDataStore.httpCookieStore
      setCookie:cookie
      completionHandler:^{
        completion(context, 0);
      }];
  return 0;
}

int rcp_https_trust_install(const char *fingerprint_hex, void *webview,
                           const char *start_url, int reset_cookies) {
  if (start_url == NULL) {
    return 4;
  }
  int installed = rcp_https_trust_install_pin(fingerprint_hex, webview);
  if (installed != 0) {
    return installed;
  }

  // Start the first HTTPS load here so nothing can request the origin before
  // the pin is in place.
  WKWebView *view = (__bridge WKWebView *)webview;
  NSURL *destination = [NSURL URLWithString:[NSString stringWithUTF8String:start_url]];
  if (destination == nil) {
    return 6;
  }
  NSString *destination_text = destination.absoluteString;
  NSURLRequest *request = [NSURLRequest requestWithURL:destination];
  if (reset_cookies) {
    // The login phase must begin with no stored session, or a cookie left by an
    // earlier run would be mistaken for a fresh one.
    NSSet *types = [NSSet setWithObject:WKWebsiteDataTypeCookies];
    [view.configuration.websiteDataStore
        removeDataOfTypes:types
            modifiedSince:[NSDate distantPast]
        completionHandler:^{
          fprintf(stderr, "[https-trust] cleared stored cookies before the drive\n");
          [view loadRequest:request];
          fprintf(stderr, "[https-trust] started load of %s\n",
                  destination_text.UTF8String ?: "<unknown>");
        }];
    return 0;
  }
  [view loadRequest:request];
  fprintf(stderr, "[https-trust] started load of %s\n",
          destination_text.UTF8String ?: "<unknown>");
  return 0;
}

void rcp_https_trust_stats(int *accepted, int *refused, int *delegated) {
  if (accepted != NULL) {
    *accepted = gAccepted;
  }
  if (refused != NULL) {
    *refused = gRefused;
  }
  if (delegated != NULL) {
    *delegated = gDelegated;
  }
}
