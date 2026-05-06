<template>
  <div class="page-bg">
    <div class="grid-overlay"></div>
    <main class="container layout">
      <header class="top panel">
        <div class="brand">
          <p class="brand-en">INDUSTRIAL TOOLBOX AI</p>
          <h1>工具箱缺失检测中控台</h1>
          <p class="sub">拍照检测 / 本地导入 / OCR 编号识别 / 缺失报警</p>
        </div>
        <div class="top-right">
          <div class="clock">{{ nowText }}</div>
          <div class="lights">
            <span :class="['light', busy ? 'on-warn' : 'on-ok']"></span>
            <span :class="['light', hasAnyMissing ? 'on-warn' : 'off']"></span>
            <span
              :class="['light', detections.length > 0 ? 'on-main' : 'off']"
            ></span>
          </div>
        </div>
      </header>

      <section class="toolbar panel">
        <div class="ctrl">
          <label>摄像头索引</label>
          <input v-model.number="cameraIndex" type="number" min="0" />
        </div>
        <div class="ctrl actions">
          <button class="btn main" :disabled="busy" @click="runCamera">
            拍照并检测
          </button>
          <label class="file-pick">
            <input type="file" accept="image/*" @change="onChooseFile" />
            <span>{{ selectedFile ? selectedFile.name : "选择图片文件" }}</span>
          </label>
          <button class="btn" :disabled="busy" @click="runUpload">
            导入图片检测
          </button>
        </div>
      </section>

      <section class="stats">
        <article class="panel stat">
          <p>系统状态</p>
          <h3 :class="busy ? 't-warn' : 't-ok'">
            {{ busy ? "检测中" : "待机" }}
          </h3>
          <small>{{ busy ? "模型推理进行中" : "等待操作" }}</small>
        </article>
        <article class="panel stat">
          <p>检测目标数</p>
          <h3>{{ detectionCount }}</h3>
          <small>当前推理框数量</small>
        </article>
        <article class="panel stat">
          <p>缺失总数</p>
          <h3 :class="missingTotalItems > 0 ? 't-warn' : 't-ok'">
            {{ missingTotalItems }}
          </h3>
          <small
            >数量缺失: {{ missingTotalItems }} / 编号缺失:
            {{ missingTotalCodes }}</small
          >
        </article>
        <article class="panel stat">
          <p>检测耗时</p>
          <h3>{{ processingCost }}</h3>
          <small>检测: {{ detectCost }} / OCR: {{ ocrCost }}</small>
          <small>OCR状态: {{ ocrStatusText }}</small>
        </article>
      </section>

      <section v-if="alarmText" :class="['alarm', alarmWarn ? 'warn' : 'ok']">
        <strong>{{ alarmWarn ? "报警" : "提示" }}</strong>
        <span>{{ alarmText }}</span>
      </section>

      <section class="images">
        <article class="panel image-card">
          <div class="card-head">
            <h2>输入图像</h2>
            <span class="tag">Source</span>
          </div>
          <div class="image-box">
            <img
              v-if="inputImageUrl"
              :src="inputImageUrl"
              alt="capture-live"
              @error="onInputImageError"
              @load="onInputImageLoad"
            />
            <p v-else>尚未输入图像</p>
            <div v-if="busy" class="detecting-mask">正在检测中...</div>
            <div class="scan-line"></div>
          </div>
        </article>

        <article class="panel image-card">
          <div class="card-head">
            <h2>推理结果图（含检测框）</h2>
            <span class="tag">Inference</span>
          </div>
          <div class="image-box">
            <img
              v-if="annotatedImageUrl"
              :src="annotatedImageUrl"
              alt="annotated"
            />
            <p v-else>尚无检测结果</p>
            <div class="scan-line"></div>
          </div>
        </article>
      </section>

      <section class="panel table-card">
        <h2>缺失明细</h2>
        <table class="table">
          <thead>
            <tr>
              <th>零件</th>
              <th>缺失数量</th>
              <th>缺失编号</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="missingParts.length === 0">
              <td colspan="3" class="t-ok">未检测到缺失</td>
            </tr>
            <tr
              v-for="item in missingParts"
              :key="`${item.class_name}-${item.missing_count}`"
            >
              <td>{{ item.class_name || "-" }}</td>
              <td class="t-warn">{{ item.missing_count }}</td>
              <td>
                {{
                  item.missing_codes?.length
                    ? item.missing_codes.join(", ")
                    : item.missing_code_hint || "-"
                }}
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="panel table-card">
        <h2>零件统计与 OCR</h2>
        <table class="table">
          <thead>
            <tr>
              <th>零件</th>
              <th>应有</th>
              <th>检测到</th>
              <th>缺失</th>
              <th>OCR 文本</th>
              <th>OCR 编号</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="partSummaries.length === 0">
              <td colspan="6" class="t-muted">暂无数据</td>
            </tr>
            <tr v-for="row in partSummaries" :key="row.class_name">
              <td>{{ row.class_name }}</td>
              <td>{{ row.required_count }}</td>
              <td>{{ row.found_count }}</td>
              <td :class="row.missing_count > 0 ? 't-warn' : 't-ok'">
                {{ row.missing_count }}
              </td>
              <td>
                {{ row.ocr_texts?.length ? row.ocr_texts.join(" | ") : "-" }}
              </td>
              <td>
                {{ row.ocr_codes?.length ? row.ocr_codes.join(", ") : "-" }}
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="panel table-card">
        <h2>检测项 OCR 明细</h2>
        <table class="table">
          <thead>
            <tr>
              <th>#</th>
              <th>Slot</th>
              <th>Expected</th>
              <th>Recognized</th>
              <th>Status</th>
              <th>Confidence</th>
              <th>OCR Crop</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="slotResults.length === 0">
              <td colspan="7" class="t-muted">暂无数据</td>
            </tr>
            <tr
              v-for="(slot, idx) in slotResults"
              :key="slot.slot_code || idx"
            >
              <td>{{ idx + 1 }}</td>
              <td>{{ slot.slot_code || "-" }}</td>
              <td>{{ slot.expected_code || slot.slot_code || "-" }}</td>
              <td>{{ slot.recognized_code || "-" }}</td>
              <td :class="slotStatusClass(slot)">
                {{ slotStatusText(slot) }}
              </td>
              <td>
                {{
                  typeof slot.detection?.confidence === "number"
                    ? slot.detection.confidence.toFixed(3)
                    : "-"
                }}
              </td>
              <td>
                <img
                  v-if="slotOcrCropUrl(slot)"
                  class="ocr-crop"
                  :src="slotOcrCropUrl(slot)"
                  alt="OCR crop"
                />
                <span v-else>-</span>
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from "vue";

const busy = ref(false);
const cameraIndex = ref(0);
const selectedFile = ref(null);

const nowText = ref("");
const processingCost = ref("-");
const detectCost = ref("-");
const ocrCost = ref("-");
const detectionCount = ref(0);
const missingTotalItems = ref(0);
const missingTotalCodes = ref(0);
const ocrStatusText = ref("unknown");

const alarmText = ref("");
const alarmWarn = ref(false);

const captureImageUrl = ref("");
const annotatedImageUrl = ref("");
const missingParts = ref([]);
const partSummaries = ref([]);
const detections = ref([]);
const slotResults = ref([]);
const frozenInputUrl = ref("");
const selectedImagePreviewUrl = ref("");

let timer = null;
let previewTimer = null;
let streamRetryTimer = null;
let asyncResultTimer = null;
let streamLoadWatchdogTimer = null;
const liveStreamUrl = ref("");
const useSnapshotPreview = ref(false);
const previewInFlight = ref(false);
const streamBroken = ref(false);
const pendingAsyncTaskId = ref("");
const apiBase = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");
const CAMERA_DETECT_TIMEOUT_MS = 60000;
const UPLOAD_DETECT_TIMEOUT_MS = 60000;
const FORCE_PREVIEW_POLLING = false;
const PREVIEW_POLL_INTERVAL_MS = 1500;
const hasAnyMissing = computed(
  () => missingTotalItems.value > 0 || missingTotalCodes.value > 0,
);

function apiUrl(path) {
  if (!apiBase) return path;
  return `${apiBase}${path}`;
}

function withTs(url) {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}t=${Date.now()}`;
}

const inputImageUrl = computed(() => {
  if (FORCE_PREVIEW_POLLING && !useSnapshotPreview.value) {
    if (selectedImagePreviewUrl.value) return selectedImagePreviewUrl.value;
    if (frozenInputUrl.value) return frozenInputUrl.value;
    if (captureImageUrl.value) return captureImageUrl.value;
    return "";
  }
  // Realtime mode: always prioritize live stream on the left panel.
  if (!useSnapshotPreview.value && liveStreamUrl.value && !streamBroken.value)
    return liveStreamUrl.value;
  if (selectedImagePreviewUrl.value) return selectedImagePreviewUrl.value;
  if (frozenInputUrl.value) return frozenInputUrl.value;
  if (liveStreamUrl.value && !streamBroken.value) return liveStreamUrl.value;
  if (captureImageUrl.value) return captureImageUrl.value;
  return "";
});

async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timerId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    let data = {};
    try {
      data = await res.json();
    } catch (_e) {
      data = {};
    }
    return { res, data };
  } finally {
    window.clearTimeout(timerId);
  }
}

function updateClock() {
  const d = new Date();
  nowText.value = d.toLocaleString("zh-CN", { hour12: false });
}

function playWebAlarm() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 1040;
    gain.gain.value = 0.15;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    setTimeout(() => osc.stop(), 280);
  } catch (e) {
    console.log("audio failed", e);
  }
}

function resetAlarm() {
  alarmText.value = "";
  alarmWarn.value = false;
}

function slotStatusText(slot) {
  if (!slot?.found) return "Missing";
  return slot.slot_match ? "OK" : "Mismatch";
}

function slotStatusClass(slot) {
  return slot?.found && slot.slot_match ? "t-ok" : "t-warn";
}

function slotOcrCropUrl(slot) {
  const url = slot?.detection?.ocr_crop_url;
  return url ? withTs(apiUrl(url)) : "";
}

function applyResult(data) {
  captureImageUrl.value = data.capture_image_url
    ? withTs(apiUrl(data.capture_image_url))
    : "";
  annotatedImageUrl.value = data.annotated_image_url
    ? withTs(apiUrl(data.annotated_image_url))
    : "";

  missingParts.value = data.missing_parts || [];
  partSummaries.value = data.part_summaries || [];
  detections.value = data.detections || [];
  slotResults.value = data.slot_results || [];

  detectionCount.value = data.detection_count ?? 0;
  missingTotalItems.value = data.missing_total_items ?? 0;
  missingTotalCodes.value = data.missing_total_codes ?? 0;
  ocrStatusText.value = String(data.ocr_status || "unknown");

  const ms = data.processing_time_ms ?? 0;
  const sec = data.processing_time_sec ?? ms / 1000;
  processingCost.value = `${ms} ms / ${Number(sec).toFixed(3)} s`;
  const t = data.timings_ms || {};
  detectCost.value =
    typeof t.detect === "number" ? `${Math.round(t.detect)} ms` : "-";
  ocrCost.value = typeof t.ocr === "number" ? `${Math.round(t.ocr)} ms` : "-";

  if (ocrStatusText.value === "skipped_fast_stage") {
    alarmWarn.value = false;
    alarmText.value = "快速检测完成，当前仅显示检测框。";
  } else if (data.missing_status === "missing_detected") {
    alarmWarn.value = true;
    alarmText.value = `检测到缺失：数量 ${missingTotalItems.value} 个，编号 ${missingTotalCodes.value} 个，请立即复核。`;
    playWebAlarm();
  } else if (ocrStatusText.value.toLowerCase().startsWith("failed")) {
    alarmWarn.value = true;
    alarmText.value = `OCR初始化失败：${ocrStatusText.value}，当前仅显示检测框，不会有OCR文本。`;
  } else {
    alarmWarn.value = false;
    alarmText.value = "状态正常，未检测到缺失。";
  }
}

function onChooseFile(e) {
  if (selectedImagePreviewUrl.value) {
    URL.revokeObjectURL(selectedImagePreviewUrl.value);
    selectedImagePreviewUrl.value = "";
  }
  selectedFile.value = e.target.files?.[0] || null;
  if (selectedFile.value) {
    selectedImagePreviewUrl.value = URL.createObjectURL(selectedFile.value);
    frozenInputUrl.value = selectedImagePreviewUrl.value;
  }
}

function resetLiveStream() {
  const idx = Number(cameraIndex.value || 0);
  streamBroken.value = false;
  liveStreamUrl.value = withTs(apiUrl(`/api/live-stream?camera_index=${idx}`));
  if (streamLoadWatchdogTimer) {
    window.clearTimeout(streamLoadWatchdogTimer);
    streamLoadWatchdogTimer = null;
  }
  // Some browsers keep MJPEG request pending even when no frame is produced.
  // If no frame loads quickly, force fallback mode.
  streamLoadWatchdogTimer = window.setTimeout(() => {
    if (!useSnapshotPreview.value && liveStreamUrl.value) {
      streamBroken.value = true;
    }
  }, 1800);
}

function stopLiveStream() {
  liveStreamUrl.value = "";
}

async function onInputImageError() {
  if (streamLoadWatchdogTimer) {
    window.clearTimeout(streamLoadWatchdogTimer);
    streamLoadWatchdogTimer = null;
  }
  if (!liveStreamUrl.value) return;
  streamBroken.value = true;
  startStreamRetryTimer();
}

function onInputImageLoad() {
  if (streamLoadWatchdogTimer) {
    window.clearTimeout(streamLoadWatchdogTimer);
    streamLoadWatchdogTimer = null;
  }
  if (liveStreamUrl.value) {
    streamBroken.value = false;
    stopStreamRetryTimer();
  }
}

function stopPreviewTimer() {
  if (previewTimer) {
    window.clearInterval(previewTimer);
    previewTimer = null;
  }
}

function startPreviewFallbackTimer() {
  stopPreviewTimer();
  previewTimer = window.setInterval(
    async () => {
      if (useSnapshotPreview.value) return;
      if (busy.value) return;
      if (FORCE_PREVIEW_POLLING) {
        await primeInputFrame(900);
        return;
      }
      if (!streamBroken.value) return;
      await primeInputFrame(1000);
    },
    FORCE_PREVIEW_POLLING ? PREVIEW_POLL_INTERVAL_MS : 700,
  );
}

function stopStreamRetryTimer() {
  if (streamRetryTimer) {
    window.clearInterval(streamRetryTimer);
    streamRetryTimer = null;
  }
  if (streamLoadWatchdogTimer) {
    window.clearTimeout(streamLoadWatchdogTimer);
    streamLoadWatchdogTimer = null;
  }
}

function stopAsyncResultTimer() {
  if (asyncResultTimer) {
    window.clearInterval(asyncResultTimer);
    asyncResultTimer = null;
  }
  pendingAsyncTaskId.value = "";
}

function startAsyncResultPolling(taskId) {
  stopAsyncResultTimer();
  if (!taskId) return;
  pendingAsyncTaskId.value = String(taskId);
  asyncResultTimer = window.setInterval(async () => {
    if (!pendingAsyncTaskId.value) return;
    try {
      const { res, data } = await fetchJsonWithTimeout(
        apiUrl(`/api/async-result/${pendingAsyncTaskId.value}`),
        {},
        4000,
      );
      if (!res.ok) return;
      if (data?.status === "done" && data?.result) {
        applyResult(data.result);
        stopAsyncResultTimer();
      } else if (data?.status === "error") {
        alarmWarn.value = true;
        alarmText.value = `错误：${data.error || "OCR 异步结果失败"}`;
        stopAsyncResultTimer();
      }
    } catch (_e) {
      // keep polling
    }
  }, 500);
}

function startStreamRetryTimer() {
  stopStreamRetryTimer();
  streamRetryTimer = window.setInterval(() => {
    if (useSnapshotPreview.value) return;
    if (!streamBroken.value) return;
    resetLiveStream();
  }, 1200);
}

async function primeInputFrame(timeoutMs = 2500) {
  if (previewInFlight.value) return false;
  previewInFlight.value = true;
  try {
    const idx = Number(cameraIndex.value || 0);
    const { res, data } = await fetchJsonWithTimeout(
      apiUrl("/api/live-preview"),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_index: idx }),
      },
      timeoutMs,
    );
    if (res.ok && data?.capture_image_url) {
      captureImageUrl.value = withTs(apiUrl(data.capture_image_url));
      return true;
    }
    return false;
  } catch (_e) {
    // ignore preview bootstrap failures; stream will keep retrying.
    return false;
  } finally {
    previewInFlight.value = false;
  }
}

function startPreview() {
  stopPreviewTimer();
  if (useSnapshotPreview.value) {
    stopStreamRetryTimer();
    stopLiveStream();
    // In hik_script mode, never auto-capture in background.
    // Capture only when user explicitly clicks detect.
    return;
  }
  if (FORCE_PREVIEW_POLLING) {
    stopStreamRetryTimer();
    stopLiveStream();
    // Continuous preview via live-preview polling.
    startPreviewFallbackTimer();
    primeInputFrame(800);
    return;
  }
  resetLiveStream();
  startStreamRetryTimer();
  startPreviewFallbackTimer();
}

function handleVisibilityChange() {
  if (document.hidden) {
    stopPreviewTimer();
    stopStreamRetryTimer();
    stopLiveStream();
    return;
  }
  startPreview();
}

async function runCamera() {
  stopAsyncResultTimer();
  busy.value = true;
  resetAlarm();
  stopPreviewTimer();
  stopStreamRetryTimer();
  stopLiveStream();
  if (useSnapshotPreview.value) {
    for (let i = 0; i < 20 && previewInFlight.value; i += 1) {
      // Wait briefly for an inflight preview capture to finish to avoid camera contention.
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }
  }
  try {
    const idx = Number(cameraIndex.value || 0);
    if (useSnapshotPreview.value) {
      // Snapshot mode: best-effort frame bootstrap.
      const previewTimeout = 1200;
      const { res: pRes, data: pData } = await fetchJsonWithTimeout(
        apiUrl("/api/live-preview"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ camera_index: idx }),
        },
        previewTimeout,
      );
      if (pRes.ok && pData?.capture_image_url) {
        const frameUrl = withTs(apiUrl(pData.capture_image_url));
        captureImageUrl.value = frameUrl;
        frozenInputUrl.value = frameUrl;
      } else if (captureImageUrl.value) {
        frozenInputUrl.value = captureImageUrl.value;
      }
    }
  } catch (_e) {
    if (captureImageUrl.value) {
      frozenInputUrl.value = captureImageUrl.value;
    }
  }
  try {
    const { res, data } = await fetchJsonWithTimeout(
      apiUrl("/api/capture-and-detect"),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_index: Number(cameraIndex.value || 0) }),
      },
      CAMERA_DETECT_TIMEOUT_MS,
    );
    if (!res.ok) throw new Error(data.detail || "摄像头检测失败");
    applyResult(data);
    if (data.async_task_id) {
      startAsyncResultPolling(data.async_task_id);
    } else {
      stopAsyncResultTimer();
    }
  } catch (err) {
    alarmWarn.value = true;
    const isAbort = err?.name === "AbortError";
    alarmText.value = isAbort
      ? `错误：检测超时（${Math.round(CAMERA_DETECT_TIMEOUT_MS / 1000)}秒），请重试。`
      : `错误：${err.message}`;
    processingCost.value = "-";
    detectCost.value = "-";
    ocrCost.value = "-";
    playWebAlarm();
  } finally {
    busy.value = false;
    frozenInputUrl.value = "";
    startPreview();
  }
}

async function runUpload() {
  stopAsyncResultTimer();
  if (!selectedFile.value) {
    alarmWarn.value = true;
    alarmText.value = "请先选择图片文件";
    return;
  }

  busy.value = true;
  resetAlarm();
  stopPreviewTimer();
  stopStreamRetryTimer();
  stopLiveStream();
  if (selectedImagePreviewUrl.value) {
    frozenInputUrl.value = selectedImagePreviewUrl.value;
  }
  try {
    const fd = new FormData();
    fd.append("image", selectedFile.value);

    const { res, data } = await fetchJsonWithTimeout(
      apiUrl("/api/detect-upload"),
      {
        method: "POST",
        body: fd,
      },
      UPLOAD_DETECT_TIMEOUT_MS,
    );
    if (!res.ok) throw new Error(data.detail || "导入图片检测失败");
    applyResult(data);
    if (data.async_task_id) {
      startAsyncResultPolling(data.async_task_id);
    } else {
      stopAsyncResultTimer();
    }
    if (selectedImagePreviewUrl.value) {
      URL.revokeObjectURL(selectedImagePreviewUrl.value);
      selectedImagePreviewUrl.value = "";
    }
  } catch (err) {
    alarmWarn.value = true;
    const isAbort = err?.name === "AbortError";
    alarmText.value = isAbort
      ? `错误：检测超时（${Math.round(UPLOAD_DETECT_TIMEOUT_MS / 1000)}秒），请重试。`
      : `错误：${err.message}`;
    processingCost.value = "-";
    detectCost.value = "-";
    ocrCost.value = "-";
    playWebAlarm();
  } finally {
    busy.value = false;
    startPreview();
  }
}

onMounted(async () => {
  updateClock();
  timer = window.setInterval(updateClock, 1000);
  try {
    const { res, data } = await fetchJsonWithTimeout(
      apiUrl("/api/camera-mode"),
      {},
      3000,
    );
    if (res.ok && data) {
      if (typeof data.use_snapshot_preview === "boolean") {
        useSnapshotPreview.value = data.use_snapshot_preview;
      } else if (data.live_provider) {
        useSnapshotPreview.value =
          String(data.live_provider).toLowerCase() === "hik_script";
      } else if (data.provider) {
        useSnapshotPreview.value =
          String(data.provider).toLowerCase() === "hik_script";
      }
    }
  } catch (_e) {
    // keep default stream mode when probe fails
  }
  startPreview();
  document.addEventListener("visibilitychange", handleVisibilityChange);
});

onUnmounted(() => {
  if (timer) window.clearInterval(timer);
  stopPreviewTimer();
  stopStreamRetryTimer();
  stopAsyncResultTimer();
  document.removeEventListener("visibilitychange", handleVisibilityChange);
  stopLiveStream();
  if (selectedImagePreviewUrl.value) {
    URL.revokeObjectURL(selectedImagePreviewUrl.value);
    selectedImagePreviewUrl.value = "";
  }
});

watch(cameraIndex, () => {
  startPreview();
});
</script>

<style scoped>
.image-box {
  position: relative;
}

.detecting-mask {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(7, 18, 38, 0.48);
  color: #d8f0ff;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 1px;
  text-shadow: 0 0 8px rgba(0, 196, 255, 0.45);
  z-index: 4;
  pointer-events: none;
}
</style>
