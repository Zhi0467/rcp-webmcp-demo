import type { AnchorHTMLAttributes, MouseEvent, ReactNode } from "react";
import { openEpisodeReportFromLink } from "../desktopRuntime";

interface Props extends Omit<
  AnchorHTMLAttributes<HTMLAnchorElement>,
  "children" | "href" | "onClick" | "rel" | "target"
> {
  projectId: string;
  episodeId: string;
  href: string;
  children: ReactNode;
  onOpenError: (message: string) => void;
}

export function EpisodeReportLink({
  projectId,
  episodeId,
  href,
  children,
  onOpenError,
  ...anchorProps
}: Props) {
  const openReport = async (event: MouseEvent<HTMLAnchorElement>) => {
    try {
      await openEpisodeReportFromLink(event, { projectId, episodeId });
    } catch (error) {
      onOpenError(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <a {...anchorProps} href={href} target="_blank" rel="noopener noreferrer" onClick={openReport}>
      {children}
    </a>
  );
}
