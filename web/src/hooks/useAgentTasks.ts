import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  isActiveTask,
  parseDismissedTaskIds,
  projectActivityTask,
  serializeDismissedTaskIds,
  taskNotificationStorageKey,
} from "../agentTasks";
import { api } from "../api";
import type { AgentTask } from "../types";

export interface AgentTasksSnapshot {
  retryTask: AgentTask | null;
  tasks: AgentTask[];
  taskInspectorId: string | null;
  inspectedTask: AgentTask | null;
  activityTaskId: string | null;
  dismissedTaskIds: Set<string>;
}

interface UseAgentTasksOptions {
  projectId: string | null;
  reportError: (message: string) => void;
}

export function cloneAgentTasksSnapshot(snapshot: AgentTasksSnapshot): AgentTasksSnapshot {
  return {
    ...snapshot,
    tasks: [...snapshot.tasks],
    dismissedTaskIds: new Set(snapshot.dismissedTaskIds),
  };
}

export function reconcileKnownActiveTasks(
  knownActive: Map<string, AgentTask>,
  current: AgentTask[],
): AgentTask[] {
  const terminal = current.filter(
    (task) => knownActive.has(task.operation_id) && !isActiveTask(task),
  );
  for (const task of terminal) knownActive.delete(task.operation_id);
  for (const task of current) {
    if (isActiveTask(task)) knownActive.set(task.operation_id, task);
  }
  return terminal;
}

export function useAgentTasks({ projectId, reportError }: UseAgentTasksOptions) {
  const [retryTask, setRetryTask] = useState<AgentTask | null>(null);
  const [taskStarting, setTaskStarting] = useState(false);
  const [taskActionId, setTaskActionId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [taskInspectorId, setTaskInspectorId] = useState<string | null>(null);
  const [inspectedTask, setInspectedTask] = useState<AgentTask | null>(null);
  const [taskInspectorLoading, setTaskInspectorLoading] = useState(false);
  const [activityTaskId, setActivityTaskId] = useState<string | null>(null);
  const [dismissedTaskIds, setDismissedTaskIds] = useState<Set<string>>(() =>
    readDismissedTaskIds(projectId),
  );
  const taskStartLock = useRef(false);
  const knownActiveTasks = useRef(new Map<string, AgentTask>());

  const rememberActiveTasks = useCallback((nextTasks: AgentTask[]) => {
    for (const task of nextTasks) {
      if (isActiveTask(task)) knownActiveTasks.current.set(task.operation_id, task);
    }
  }, []);

  const activeTask = useMemo(() => tasks.find(isActiveTask) ?? null, [tasks]);
  const activityTask = projectActivityTask(tasks, activityTaskId);

  useEffect(() => {
    if (activityTask && (isActiveTask(activityTask) || activityTask.paused)) {
      setActivityTaskId(activityTask.operation_id);
    }
  }, [activityTask]);

  const inspectorSummary = tasks.find((task) => task.operation_id === taskInspectorId);
  const inspectorVersion = inspectorSummary?.updated_at;
  useEffect(() => {
    if (!inspectorSummary) return;
    setInspectedTask((current) =>
      current?.operation_id === inspectorSummary.operation_id
        ? { ...current, ...inspectorSummary, events: current.events }
        : current,
    );
  }, [inspectorSummary]);

  useEffect(() => {
    if (!projectId || !taskInspectorId) {
      setInspectedTask(null);
      return;
    }
    let cancelled = false;
    setTaskInspectorLoading(true);
    api<AgentTask>(`/api/projects/${encodeURIComponent(projectId)}/tasks/${taskInspectorId}`)
      .then((task) => {
        if (!cancelled) setInspectedTask(task);
      })
      .catch((error) => {
        if (!cancelled) reportError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setTaskInspectorLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [inspectorVersion, projectId, taskInspectorId]);

  const replaceTasks = useCallback(
    (nextTasks: AgentTask[]) => {
      rememberActiveTasks(nextTasks);
      setTasks(nextTasks);
    },
    [rememberActiveTasks],
  );

  const upsertTask = useCallback(
    (task: AgentTask) => {
      rememberActiveTasks([task]);
      setTasks((current) => [
        task,
        ...current.filter((item) => item.operation_id !== task.operation_id),
      ]);
      setDismissedTaskIds((current) => {
        const next = new Set(current);
        next.delete(task.operation_id);
        return next;
      });
    },
    [rememberActiveTasks],
  );

  const recordStartedTask = useCallback(
    (task: AgentTask) => {
      rememberActiveTasks([task]);
      setTasks((current) => [
        task,
        ...current.filter((item) => item.operation_id !== task.operation_id),
      ]);
      setActivityTaskId(task.operation_id);
      setDismissedTaskIds((current) => {
        const next = new Set(current);
        next.delete(task.operation_id);
        return next;
      });
    },
    [rememberActiveTasks],
  );

  const consumeTerminalTasks = useCallback(
    (nextTasks: AgentTask[]) => reconcileKnownActiveTasks(knownActiveTasks.current, nextTasks),
    [],
  );

  const presentTask = useCallback((task: AgentTask) => {
    setActivityTaskId(task.operation_id);
    setTaskInspectorId(task.operation_id);
    setInspectedTask(task);
  }, []);

  const selectTaskInspector = useCallback((operationId: string | null) => {
    setTaskInspectorId(operationId);
  }, []);

  const chooseRetryTask = useCallback((task: AgentTask) => {
    setRetryTask(task);
  }, []);

  const closeRetryTask = useCallback(() => {
    setRetryTask(null);
  }, []);

  const dismissTaskNotification = useCallback(
    (operationId: string) => {
      setDismissedTaskIds((current) => {
        const next = new Set(current);
        next.add(operationId);
        try {
          localStorage.setItem(
            taskNotificationStorageKey(projectId),
            serializeDismissedTaskIds(next),
          );
        } catch {}
        return next;
      });
      if (taskInspectorId === operationId) {
        setTaskInspectorId(null);
        setInspectedTask(null);
      }
    },
    [projectId, taskInspectorId],
  );

  const beginTaskStart = useCallback(() => {
    if (taskStartLock.current || taskStarting) return null;
    taskStartLock.current = true;
    setTaskStarting(true);
    return () => {
      taskStartLock.current = false;
      setTaskStarting(false);
    };
  }, [taskStarting]);

  const beginTaskAction = useCallback(
    (operationId: string) => {
      if (taskActionId) return null;
      setTaskActionId(operationId);
      return () => setTaskActionId(null);
    },
    [taskActionId],
  );

  const beginTaskRepair = useCallback(
    (operationId: string) => {
      if (taskStartLock.current || taskStarting || taskActionId) return null;
      taskStartLock.current = true;
      setTaskStarting(true);
      setTaskActionId(operationId);
      return () => {
        taskStartLock.current = false;
        setTaskStarting(false);
        setTaskActionId(null);
      };
    },
    [taskActionId, taskStarting],
  );

  const resetProjectTasks = useCallback((nextProjectId: string | null) => {
    knownActiveTasks.current.clear();
    setRetryTask(null);
    setTasks([]);
    setTaskInspectorId(null);
    setInspectedTask(null);
    setActivityTaskId(null);
    setDismissedTaskIds(readDismissedTaskIds(nextProjectId));
  }, []);

  const restoreProjectTasks = useCallback(
    (snapshot: AgentTasksSnapshot) => {
      knownActiveTasks.current.clear();
      rememberActiveTasks(snapshot.tasks);
      setRetryTask(snapshot.retryTask);
      setTasks([...snapshot.tasks]);
      setTaskInspectorId(snapshot.taskInspectorId);
      setInspectedTask(snapshot.inspectedTask);
      setActivityTaskId(snapshot.activityTaskId);
      setDismissedTaskIds(new Set(snapshot.dismissedTaskIds));
    },
    [rememberActiveTasks],
  );

  const snapshot = useMemo<AgentTasksSnapshot>(
    () => ({
      retryTask,
      tasks,
      taskInspectorId,
      inspectedTask,
      activityTaskId,
      dismissedTaskIds,
    }),
    [activityTaskId, dismissedTaskIds, inspectedTask, retryTask, taskInspectorId, tasks],
  );

  return {
    snapshot,
    taskStarting,
    taskActionId,
    taskInspectorLoading,
    activeTask,
    activityTask,
    replaceTasks,
    consumeTerminalTasks,
    upsertTask,
    recordStartedTask,
    presentTask,
    selectTaskInspector,
    chooseRetryTask,
    closeRetryTask,
    dismissTaskNotification,
    beginTaskStart,
    beginTaskAction,
    beginTaskRepair,
    resetProjectTasks,
    restoreProjectTasks,
  };
}

function readDismissedTaskIds(projectId: string | null): Set<string> {
  try {
    return parseDismissedTaskIds(localStorage.getItem(taskNotificationStorageKey(projectId)));
  } catch {
    return new Set();
  }
}
