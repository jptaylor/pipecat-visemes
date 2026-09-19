import { useSyncExternalStore } from "react";

function subscribeToFrames(notify: () => void): () => void {
  let raf = requestAnimationFrame(function tick() {
    notify();
    raf = requestAnimationFrame(tick);
  });
  return () => cancelAnimationFrame(raf);
}

/**
 * Polls `compute` every animation frame and re-renders only when its result
 * changes (compared with Object.is, so return a primitive) — e.g. the index
 * of the word under a playhead that moves every frame.
 */
export function useAnimationFrameValue<T>(compute: () => T): T {
  return useSyncExternalStore(subscribeToFrames, compute);
}
