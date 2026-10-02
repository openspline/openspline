/** Preserve contiguous audio; only prebuffer at startup or after a real underrun. */
export function nextPlayoutTime(now, previousEnd, bufferSeconds = 0.16) {
    return previousEnd >= now + 0.005 ? previousEnd : now + bufferSeconds;
}
