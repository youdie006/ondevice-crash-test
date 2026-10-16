(function () {
  "use strict";

  const STEP_ORDER = ["evaluating", "diagnosing", "repairing", "re-evaluating", "gate"];
  const TERMINAL_PHASES = new Set(["done", "needs_human", "error"]);
  const ACTIVE_PHASES = new Set(["starting", "evaluating", "diagnosing", "repairing", "re-evaluating", "gate"]);
  const VERIFIED_INDOOR_STEMS = new Set(["000000000139", "000000004495", "000000012576", "b_001"]);

  const urlParams = new URLSearchParams(window.location.search);
  const isMockMode = urlParams.get("mock") === "1";
  const isReplayMode = urlParams.get("replay") === "showcase";
  const currentScenario = urlParams.get("scenario") || "robot_vacuum";

  const prefersReducedMotion =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let currentState = null;
  let activeRunId = null;
  let selectedLiveCondition = null;
  let selectedInspectCondition = null;
  let userPickedInspectCondition = false;

  // Tracking state for honest event-driven motion
  let seenRecentKeys = new Set();
  let previousCardF1 = {};
  let previousCardPass = {};
  let previousControlF1 = null;
  let lastDiagnosisSignature = "";
  let lastGateSignature = "";
  let lastPipelineEventKey = "";
  let currentPipelineHopIndex = 0;

  // Observed phase durations & live timer state
  const observedPhaseDurations = {};
  let timerAnchorPhase = null;
  let timerAnchorStartedAtMs = null;
  let timerAnchorBaseElapsedS = null;
  let timerAnchorClientPerfMs = null;

  let argonAnchorStatus = null;
  let argonAnchorElapsedS = null;
  let argonAnchorClientPerfMs = null;

  // Mock / Replay state (strictly isolated to client memory)
  let mockSequence = [];
  let mockStepIndex = 0;
  let mockIntervalId = null;

  function isFiniteNum(v) {
    return typeof v === "number" && Number.isFinite(v);
  }

  function fmtF1(v) {
    return isFiniteNum(v) ? v.toFixed(4) : "N/A";
  }

  function fmtNum(v, digits) {
    return isFiniteNum(v) ? v.toFixed(digits) : "N/A";
  }

  function fmtInt(v) {
    return typeof v === "number" && Number.isFinite(v) ? String(Math.round(v)) : "N/A";
  }

  function fmtMMSS(seconds) {
    if (!isFiniteNum(seconds) || seconds < 0) return "N/A";
    const total = Math.floor(seconds);
    const mm = String(Math.floor(total / 60)).padStart(2, "0");
    const ss = String(total % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }

  function fmtSec(seconds) {
    if (!isFiniteNum(seconds) || seconds < 0) return "—";
    return `${seconds.toFixed(1)}s`;
  }

  function normStem(idStr) {
    if (!idStr) return "";
    const base = String(idStr).split("/").pop() || "";
    return base.replace(/\.[^.]+$/, "");
  }

  function isIndoorPriorityItem(item) {
    if (!item || typeof item !== "object") return false;
    if (item.has_indoor_objects === true) return true;
    const stem = normStem(item.image_id || item.file_name || "");
    const srcStem = normStem(item.source_image_id || "");
    return VERIFIED_INDOOR_STEMS.has(stem) || VERIFIED_INDOOR_STEMS.has(srcStem);
  }

  function getRawPassThreshold(state, cstate) {
    if (cstate && typeof cstate === "object") {
      if (isFiniteNum(cstate.pass_threshold_raw)) return cstate.pass_threshold_raw;
      if (isFiniteNum(cstate.pass_threshold)) return cstate.pass_threshold;
    }
    if (state && typeof state === "object") {
      if (isFiniteNum(state.pass_threshold_raw)) return state.pass_threshold_raw;
      if (isFiniteNum(state.pass_threshold)) return state.pass_threshold;
      const bc = state.baseline_control;
      if (bc && typeof bc === "object") {
        if (isFiniteNum(bc.raw_f1)) return bc.raw_f1 - 0.05;
        if (isFiniteNum(bc.f1)) return bc.f1 - 0.05;
      }
    }
    return null;
  }

  function getDisplayPassThreshold(state, cstate) {
    if (cstate && typeof cstate === "object" && isFiniteNum(cstate.pass_threshold)) {
      return cstate.pass_threshold;
    }
    if (state && typeof state === "object" && isFiniteNum(state.pass_threshold)) {
      return state.pass_threshold;
    }
    const raw = getRawPassThreshold(state, cstate);
    return isFiniteNum(raw) ? Number(raw.toFixed(4)) : null;
  }

  function getControlSanityThreshold(state) {
    if (state && isFiniteNum(state.control_sanity_threshold)) {
      return state.control_sanity_threshold;
    }
    return 0.80;
  }

  function evaluateControlPass(ctrlObj, sanityThreshold) {
    if (!ctrlObj || typeof ctrlObj !== "object") return null;
    const raw = isFiniteNum(ctrlObj.raw_f1) ? ctrlObj.raw_f1 : (isFiniteNum(ctrlObj.f1) ? ctrlObj.f1 : null);
    if (raw === null) return null;
    if (typeof ctrlObj.passed === "boolean") {
      return ctrlObj.passed && raw >= sanityThreshold;
    }
    return raw >= sanityThreshold;
  }

  function extractConditionMetrics(cstate, state) {
    if (!cstate || typeof cstate !== "object") {
      return {
        hasMeasured: false,
        baselineF1: null,
        finalF1: null,
        baselineRawF1: null,
        finalRawF1: null,
        activeF1: null,
        activeRawF1: null,
        isPass: false,
        isLimitation: false,
        refObjects: null,
        tp: null,
        fp: null,
        fn: null,
        retention: null,
        nImages: null,
        validImages: null,
        statusLabel: "evaluation pending"
      };
    }

    const bm = cstate.baseline_metrics && typeof cstate.baseline_metrics === "object" ? cstate.baseline_metrics : null;
    const fm = cstate.final_metrics && typeof cstate.final_metrics === "object" ? cstate.final_metrics : null;

    const baselineF1 = isFiniteNum(cstate.baseline_f1)
      ? cstate.baseline_f1
      : (bm && isFiniteNum(bm.f1) ? bm.f1 : (isFiniteNum(cstate.f1) ? cstate.f1 : null));

    const finalF1 = isFiniteNum(cstate.final_f1)
      ? cstate.final_f1
      : (fm && isFiniteNum(fm.f1) ? fm.f1 : null);

    const baselineRawF1 = isFiniteNum(cstate.baseline_raw_f1)
      ? cstate.baseline_raw_f1
      : (bm && isFiniteNum(bm.raw_f1) ? bm.raw_f1 : (isFiniteNum(cstate.raw_f1) ? cstate.raw_f1 : baselineF1));

    const finalRawF1 = isFiniteNum(cstate.final_raw_f1)
      ? cstate.final_raw_f1
      : (fm && isFiniteNum(fm.raw_f1) ? fm.raw_f1 : finalF1);

    const hasMeasured = isFiniteNum(baselineF1) || isFiniteNum(finalF1);
    const activeF1 = isFiniteNum(finalF1) ? finalF1 : baselineF1;
    const activeRawF1 = isFiniteNum(finalRawF1) ? finalRawF1 : baselineRawF1;

    const rawThresh = getRawPassThreshold(state, cstate);
    let isPass = false;
    if (hasMeasured) {
      if (fm && typeof fm.passed === "boolean" && isFiniteNum(finalRawF1)) {
        isPass = fm.passed;
      } else if (isFiniteNum(activeRawF1) && isFiniteNum(rawThresh)) {
        isPass = activeRawF1 >= rawThresh;
      } else if (typeof cstate.passed === "boolean") {
        isPass = cstate.passed;
      }
    }

    const isLimitation = Boolean(
      cstate.limitation === true ||
      (bm && bm.limitation === true) ||
      (fm && fm.limitation === true)
    );

    const activeMetricsObj = fm || bm || cstate;
    const refObjects = isFiniteNum(cstate.ref_objects)
      ? cstate.ref_objects
      : (isFiniteNum(activeMetricsObj.ref_objects) ? activeMetricsObj.ref_objects : null);

    const tp = isFiniteNum(activeMetricsObj.tp) ? activeMetricsObj.tp : null;
    const fp = isFiniteNum(activeMetricsObj.fp) ? activeMetricsObj.fp : null;
    const fn = isFiniteNum(activeMetricsObj.fn) ? activeMetricsObj.fn : null;
    const retention = isFiniteNum(activeMetricsObj.retention) ? activeMetricsObj.retention : null;
    const nImages = isFiniteNum(activeMetricsObj.n_images) ? activeMetricsObj.n_images : null;
    const validImages = isFiniteNum(activeMetricsObj.valid_images) ? activeMetricsObj.valid_images : null;

    // Compression pass/fail applies to EVERY condition (Block C 14:08 rule amendment)
    let statusLabel = "evaluation pending";
    if (hasMeasured) {
      statusLabel = isPass ? "PASS" : "FAIL";
    }

    return {
      hasMeasured,
      baselineF1,
      finalF1,
      baselineRawF1,
      finalRawF1,
      activeF1,
      activeRawF1,
      isPass,
      isLimitation,
      refObjects,
      tp,
      fp,
      fn,
      retention,
      nImages,
      validImages,
      statusLabel
    };
  }

  function animateValueText(el, fromVal, toVal, durationMs, prefix, suffix) {
    if (!el) return;
    const pre = prefix || "";
    const suf = suffix || "";
    if (prefersReducedMotion || !isFiniteNum(fromVal) || !isFiniteNum(toVal) || Math.abs(fromVal - toVal) < 1e-6) {
      el.textContent = `${pre}${fmtF1(toVal)}${suf}`;
      return;
    }
    const start = performance.now();
    function frame(now) {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      const cur = fromVal + (toVal - fromVal) * eased;
      el.textContent = `${pre}${cur.toFixed(4)}${suf}`;
      if (t < 1) {
        requestAnimationFrame(frame);
      } else {
        el.textContent = `${pre}${fmtF1(toVal)}${suf}`;
      }
    }
    requestAnimationFrame(frame);
  }

  function syncTimersFromState(state) {
    const phase = String((state.progress && state.progress.phase) || state.phase || "");
    const prog = state.progress && typeof state.progress === "object" ? state.progress : null;

    if (TERMINAL_PHASES.has(phase) || !ACTIVE_PHASES.has(phase)) {
      if (timerAnchorPhase && STEP_ORDER.includes(timerAnchorPhase) && timerAnchorClientPerfMs !== null) {
        const finalElapsed = computeLivePhaseElapsed();
        if (isFiniteNum(finalElapsed)) {
          observedPhaseDurations[timerAnchorPhase] = finalElapsed;
        }
      }
      timerAnchorPhase = null;
      timerAnchorStartedAtMs = null;
      timerAnchorBaseElapsedS = null;
      timerAnchorClientPerfMs = null;
    } else if (prog) {
      const pPhase = String(prog.phase || phase);
      if (timerAnchorPhase && timerAnchorPhase !== pPhase && STEP_ORDER.includes(timerAnchorPhase)) {
        const prevElapsed = computeLivePhaseElapsed();
        if (isFiniteNum(prevElapsed)) {
          observedPhaseDurations[timerAnchorPhase] = prevElapsed;
        }
      }
      const parsedStart = prog.phase_started_at ? Date.parse(prog.phase_started_at) : NaN;
      const hasValidStart = Number.isFinite(parsedStart);
      const baseElapsed = isFiniteNum(prog.phase_elapsed_s) ? prog.phase_elapsed_s : null;

      if (timerAnchorPhase !== pPhase || hasValidStart || baseElapsed !== null) {
        timerAnchorPhase = pPhase;
        timerAnchorStartedAtMs = hasValidStart ? parsedStart : null;
        timerAnchorBaseElapsedS = baseElapsed;
        timerAnchorClientPerfMs = performance.now();
      }
    }

    // Argon thinking timer anchor
    const ar = prog && prog.argon && typeof prog.argon === "object" ? prog.argon : null;
    if (!ar || TERMINAL_PHASES.has(phase) || !ACTIVE_PHASES.has(phase)) {
      if (ar && isFiniteNum(ar.elapsed_s)) {
        argonAnchorElapsedS = ar.elapsed_s;
      }
      argonAnchorStatus = ar ? ar.status : null;
      argonAnchorClientPerfMs = null;
    } else {
      argonAnchorStatus = ar.status || null;
      if (isFiniteNum(ar.elapsed_s)) {
        argonAnchorElapsedS = ar.elapsed_s;
        argonAnchorClientPerfMs = ar.status === "thinking" ? performance.now() : null;
      } else if (ar.status !== "thinking") {
        argonAnchorClientPerfMs = null;
      }
    }
  }

  function computeLivePhaseElapsed() {
    if (!timerAnchorPhase) return null;
    if (timerAnchorStartedAtMs !== null) {
      return Math.max(0, (Date.now() - timerAnchorStartedAtMs) / 1000);
    }
    if (timerAnchorBaseElapsedS !== null && timerAnchorClientPerfMs !== null) {
      return Math.max(0, timerAnchorBaseElapsedS + (performance.now() - timerAnchorClientPerfMs) / 1000);
    }
    return timerAnchorBaseElapsedS;
  }

  function computeLiveArgonElapsed() {
    if (argonAnchorElapsedS === null) return null;
    if (argonAnchorStatus === "thinking" && argonAnchorClientPerfMs !== null) {
      return Math.max(0, argonAnchorElapsedS + (performance.now() - argonAnchorClientPerfMs) / 1000);
    }
    return argonAnchorElapsedS;
  }

  function updateLiveTimerDOM() {
    if (!currentState) return;
    const phase = String((currentState.progress && currentState.progress.phase) || currentState.phase || "");
    const activeStepIdx = STEP_ORDER.indexOf(phase);
    const isTerminal = TERMINAL_PHASES.has(phase);

    STEP_ORDER.forEach((stepKey, idx) => {
      const timerEl = document.querySelector(`[data-step-timer="${stepKey}"]`);
      if (!timerEl) return;
      if (stepKey === phase && !isTerminal) {
        const liveSec = computeLivePhaseElapsed();
        timerEl.textContent = isFiniteNum(liveSec) ? fmtSec(liveSec) : "active";
      } else if (isFiniteNum(observedPhaseDurations[stepKey])) {
        timerEl.textContent = fmtSec(observedPhaseDurations[stepKey]);
      } else if ((activeStepIdx > idx || (isTerminal && phase === "done")) && currentState.progress) {
        timerEl.textContent = "done";
      } else {
        timerEl.textContent = "—";
      }
    });

    const argonTimerEl = document.getElementById("argon-thinking-timer");
    const pipeArgonTimerEl = document.getElementById("pipeline-argon-timer");
    const liveArgonSec = computeLiveArgonElapsed();
    const formattedArgon = isFiniteNum(liveArgonSec) ? fmtMMSS(liveArgonSec) : "N/A";
    if (argonTimerEl) argonTimerEl.textContent = formattedArgon;
    if (pipeArgonTimerEl) pipeArgonTimerEl.textContent = isFiniteNum(liveArgonSec) ? fmtMMSS(liveArgonSec) : "—";
  }

  function renderTopBannersAndStatus(state) {
    const phase = String(state.phase || "not yet run");
    const statusBadge = document.getElementById("run-status-badge");
    if (statusBadge) {
      statusBadge.textContent = `Phase: ${phase}`;
    }
    const runIdBadge = document.getElementById("run-id-badge");
    if (runIdBadge) {
      runIdBadge.textContent = `Run: ${state.run_id || "N/A"}`;
    }

    // Mock or Replay mode banner
    const mockBanner = document.getElementById("mock-mode-banner");
    if (mockBanner) {
      if (isMockMode) {
        mockBanner.style.display = "flex";
        mockBanner.querySelector(".mode-banner-text").textContent =
          "Mock preview — not measured (client-side simulation isolated from live /state and showcase files)";
      } else if (isReplayMode) {
        mockBanner.style.display = "flex";
        mockBanner.querySelector(".mode-banner-text").textContent =
          "Replay of showcase run (4x)";
      } else {
        mockBanner.style.display = "none";
      }
    }

    // Highlight active live condition button
    const isRunning = ACTIVE_PHASES.has(phase) || (state.progress && ACTIVE_PHASES.has(String(state.progress.phase || "")));
    if (!isRunning && TERMINAL_PHASES.has(phase)) {
      // Keep selected highlight for the completed run so judge sees which button was run, but re-enable buttons
    }
    const condBtns = document.querySelectorAll(".run-cond-btn");
    condBtns.forEach((btn) => {
      const cid = btn.getAttribute("data-live-cond");
      btn.disabled = isRunning && !isMockMode;
      if (selectedLiveCondition && cid === selectedLiveCondition) {
        btn.classList.add("selected-live-cond");
      } else {
        btn.classList.remove("selected-live-cond");
      }
    });

    // Alert banners
    const alertContainer = document.getElementById("alert-banners");
    if (!alertContainer) return;
    alertContainer.textContent = "";

    if (state.error) {
      const errDiv = document.createElement("div");
      errDiv.className = "banner banner-error";
      const strong = document.createElement("strong");
      strong.textContent = "Run Error: ";
      const span = document.createElement("span");
      span.textContent = String(state.error);
      errDiv.appendChild(strong);
      errDiv.appendChild(span);
      alertContainer.appendChild(errDiv);
    }

    if (state.needs_human || phase === "needs_human") {
      const humDiv = document.createElement("div");
      humDiv.className = "banner banner-human";

      const contentDiv = document.createElement("div");
      contentDiv.className = "banner-human-content";
      const strong = document.createElement("strong");
      strong.textContent = "Needs Human Review: ";
      contentDiv.appendChild(strong);
      contentDiv.appendChild(
        document.createTextNode(
          "Unresolved condition(s) or unverified claims require human engineering sign-off. Nothing ships automatically. "
        )
      );
      if (state.human_decision) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = `Recorded decision: ${state.human_decision}`;
        contentDiv.appendChild(badge);
      }

      const actionsDiv = document.createElement("div");
      actionsDiv.className = "banner-human-actions";
      ["approved", "rejected"].forEach((dec) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "btn-secondary";
        b.textContent = dec === "approved" ? "Approve Recipe" : "Reject Recipe";
        b.addEventListener("click", () => submitHumanDecision(dec));
        actionsDiv.appendChild(b);
      });

      humDiv.appendChild(contentDiv);
      humDiv.appendChild(actionsDiv);
      alertContainer.appendChild(humDiv);
    }
  }

  function renderCrashWall(state) {
    const cw = state.crash_wall;
    if (!cw || !Array.isArray(cw.tiles)) return;

    const imgIdEl = document.getElementById("crash-wall-image-id");
    if (imgIdEl) {
      const labelId = cw.file_name
        ? `${cw.file_name}${cw.manifest_image_id ? " (" + cw.manifest_image_id + ")" : ""}`
        : (cw.image_id || "N/A");
      imgIdEl.textContent = labelId;
    }

    const dispThresh = getDisplayPassThreshold(state, null);
    const rawThresh = getRawPassThreshold(state, null);
    const sanityThresh = getControlSanityThreshold(state);
    const ctrlF1 = state.baseline_control && typeof state.baseline_control === "object" ? state.baseline_control.f1 : null;

    const cwPassEl = document.getElementById("cw-pass-threshold");
    if (cwPassEl) cwPassEl.textContent = fmtF1(dispThresh);
    const cwCtrlEl = document.getElementById("cw-control-f1");
    if (cwCtrlEl) cwCtrlEl.textContent = fmtF1(ctrlF1);
    const cwSanityEl = document.getElementById("cw-sanity-threshold");
    if (cwSanityEl) cwSanityEl.textContent = fmtNum(sanityThresh, 2);

    const grid = document.getElementById("crash-wall-grid");
    if (!grid) return;
    grid.textContent = "";

    cw.tiles.forEach((tile) => {
      const card = document.createElement("div");
      card.className = `cw-tile${tile.is_original ? " cw-tile-original" : ""}`;
      card.setAttribute("data-tile-id", tile.id || "");

      const header = document.createElement("div");
      header.className = "cw-tile-header";
      const titleSpan = document.createElement("span");
      titleSpan.className = "cw-tile-title";
      titleSpan.textContent = tile.name || tile.id || "";
      const codeEl = document.createElement("code");
      codeEl.className = "cw-tile-code";
      codeEl.textContent = tile.manifest_image_id || tile.image_id || "";
      header.appendChild(titleSpan);
      header.appendChild(codeEl);

      const thumbWrap = document.createElement("div");
      thumbWrap.className = "cw-thumb-wrap";
      if (tile.available && tile.thumb_url && tile.valid !== false) {
        const img = document.createElement("img");
        img.className = "cw-thumb-img";
        img.src = tile.thumb_url;
        img.alt = `${tile.name || tile.id} (${tile.image_id || ""})`;
        img.loading = "lazy";
        thumbWrap.appendChild(img);
      } else {
        const pendingDiv = document.createElement("div");
        pendingDiv.className = "cw-thumb-pending";
        const idLine = document.createElement("div");
        idLine.className = "cw-pending-id";
        idLine.textContent = tile.image_id || "N/A";
        const txtLine = document.createElement("div");
        txtLine.className = "cw-pending-text";
        txtLine.textContent = tile.valid === false ? "invalid condition output" : "evaluation pending";
        pendingDiv.appendChild(idLine);
        pendingDiv.appendChild(txtLine);
        thumbWrap.appendChild(pendingDiv);
      }

      const footer = document.createElement("div");
      footer.className = "cw-tile-footer";

      const countsDiv = document.createElement("div");
      countsDiv.className = "cw-box-counts";
      if (isFiniteNum(tile.fp32_boxes) || isFiniteNum(tile.int8_boxes)) {
        let countText = `FP32 ${fmtInt(tile.fp32_boxes)} | INT8 ${fmtInt(tile.int8_boxes)}`;
        if (isFiniteNum(tile.missed)) {
          countText += ` | missed ${fmtInt(tile.missed)}`;
        }
        countsDiv.textContent = countText;
      } else {
        countsDiv.textContent = "Boxes: N/A";
      }
      footer.appendChild(countsDiv);

      if (tile.generator_label) {
        const genDiv = document.createElement("div");
        genDiv.className = "cw-gen-line";
        genDiv.textContent = tile.generator_label;
        footer.appendChild(genDiv);
      }

      const metricDiv = document.createElement("div");
      metricDiv.className = "cw-metric-line";
      const pill = document.createElement("span");

      if (tile.is_original) {
        const ctrlPass = evaluateControlPass(state.baseline_control, sanityThresh);
        if (isFiniteNum(tile.f1)) {
          pill.className = `status-pill ${ctrlPass === false ? "status-fail" : "status-pass"}`;
          pill.textContent = `Control F1 ${fmtF1(tile.f1)} (${ctrlPass === false ? "BELOW SANITY" : "PASS"})`;
        } else {
          pill.className = "status-pill status-pending";
          pill.textContent = "evaluation pending";
        }
      } else {
        const cstate = state.conditions && state.conditions[tile.id] ? state.conditions[tile.id] : null;
        const cm = extractConditionMetrics(cstate, state);
        const bF1 = isFiniteNum(tile.baseline_f1) ? tile.baseline_f1 : cm.baselineF1;
        const fF1 = isFiniteNum(tile.final_f1) ? tile.final_f1 : cm.finalF1;
        const rawVal = isFiniteNum(cm.activeRawF1) ? cm.activeRawF1 : (isFiniteNum(fF1) ? fF1 : bF1);
        if (isFiniteNum(bF1) || isFiniteNum(fF1)) {
          const passBool = isFiniteNum(rawVal) && isFiniteNum(rawThresh) ? rawVal >= rawThresh : cm.isPass;
          pill.className = `status-pill ${passBool ? "status-pass" : "status-fail"}`;
          pill.textContent = `F1 ${fmtF1(bF1)}${isFiniteNum(fF1) ? " → " + fmtF1(fF1) : ""} (${passBool ? "PASS" : "FAIL"})`;
        } else {
          pill.className = "status-pill status-pending";
          pill.textContent = "evaluation pending";
        }
      }

      metricDiv.appendChild(pill);
      footer.appendChild(metricDiv);

      card.appendChild(header);
      card.appendChild(thumbWrap);
      card.appendChild(footer);
      grid.appendChild(card);
    });
  }

  function renderSummaryStrip(state) {
    const bc = state.baseline_control && typeof state.baseline_control === "object" ? state.baseline_control : null;
    const fc = state.final_control && typeof state.final_control === "object" ? state.final_control : null;
    const dispThresh = getDisplayPassThreshold(state, null);
    const sanityThresh = getControlSanityThreshold(state);

    const headlineEl = document.getElementById("summary-headline");
    if (!bc) {
      if (headlineEl) headlineEl.textContent = "Baseline control metrics: not yet run";
      return;
    }

    const ctrlF1 = isFiniteNum(bc.f1) ? bc.f1 : null;
    const finalCtrlF1 = fc && isFiniteNum(fc.f1) ? fc.f1 : null;
    const ctrlPass = evaluateControlPass(bc, sanityThresh);

    if (headlineEl) {
      headlineEl.textContent =
        `Control F1 ${fmtF1(ctrlF1)}${isFiniteNum(finalCtrlF1) ? " → " + fmtF1(finalCtrlF1) : ""} | ` +
        `Pass threshold fixed at ${fmtF1(dispThresh)} (Control F1 − 0.05)`;
    }

    const ctrlValEl = document.getElementById("metric-control-f1");
    if (ctrlValEl) {
      const targetVal = isFiniteNum(finalCtrlF1) ? finalCtrlF1 : ctrlF1;
      if (previousControlF1 !== null && isFiniteNum(targetVal) && Math.abs(previousControlF1 - targetVal) > 1e-6) {
        const prefix = isFiniteNum(finalCtrlF1) ? `${fmtF1(ctrlF1)} → ` : "";
        animateValueText(ctrlValEl, previousControlF1, targetVal, 600, prefix, "");
      } else {
        ctrlValEl.textContent = `${fmtF1(ctrlF1)}${isFiniteNum(finalCtrlF1) ? " → " + fmtF1(finalCtrlF1) : ""}`;
      }
      if (isFiniteNum(targetVal)) {
        previousControlF1 = targetVal;
      }
      ctrlValEl.className = `metric-value ${ctrlPass === false ? "text-fail" : "text-pass"}`;
    }

    const ctrlNoteEl = document.getElementById("metric-control-note");
    if (ctrlNoteEl) {
      const ctrlN = isFiniteNum(bc.n_images) ? bc.n_images : 40;
      ctrlNoteEl.textContent =
        `Reference objects |R|: ${fmtInt(bc.ref_objects)} (${ctrlN} Split-B images) | ` +
        `Sanity bar ≥ ${fmtNum(sanityThresh, 2)} (${ctrlPass === false ? "FAIL" : "PASS"})`;
    }

    const threshValEl = document.getElementById("metric-pass-threshold");
    if (threshValEl) {
      threshValEl.textContent = isFiniteNum(dispThresh) ? `F1 ≥ ${fmtF1(dispThresh)}` : "N/A";
    }

    const matrixNoteEl = document.getElementById("metric-matrix-summary");
    if (matrixNoteEl) {
      const condKeys = state.conditions && typeof state.conditions === "object" ? Object.keys(state.conditions) : [];
      const totalConds =
        state.scenario_meta && Array.isArray(state.scenario_meta.conditions)
          ? state.scenario_meta.conditions.length
          : 7;
      let passCount = 0;
      let failCount = 0;
      condKeys.forEach((k) => {
        const cm = extractConditionMetrics(state.conditions[k], state);
        if (cm.hasMeasured) {
          if (cm.isPass) passCount += 1;
          else failCount += 1;
        }
      });
      if (condKeys.length > 0) {
        matrixNoteEl.textContent =
          `Rule: Baseline Control F1 (${fmtF1(ctrlF1)}) − 0.05. ` +
          `Measured ${passCount + failCount}/${totalConds} conditions (${passCount} pass, ${failCount} fail; run scope: ${state.n_images || "N/A"} images).`;
      } else {
        matrixNoteEl.textContent =
          `Rule: Baseline Control F1 (${fmtF1(ctrlF1)}) − 0.05. Condition evaluation pending (${totalConds} conditions).`;
      }
    }
  }

  function renderPipelineSchematic(state) {
    const phase = String((state.progress && state.progress.phase) || state.phase || "");
    const prog = state.progress && typeof state.progress === "object" ? state.progress : null;
    const isActive = ACTIVE_PHASES.has(phase);
    const isDiagnosing = phase === "diagnosing";
    const isRepairing = phase === "repairing";

    const rowEl = document.getElementById("pipeline-main-row");
    const argonNodeEl = document.getElementById("pipeline-node-argon");
    const tokenEl = document.getElementById("pipeline-photo-token");
    const tokenImgEl = document.getElementById("pipeline-token-img");
    const tokenLabelEl = document.getElementById("pipeline-token-label");
    const repairChipsEl = document.getElementById("pipeline-repair-chips");
    const repairBarEl = document.getElementById("pipeline-repair-bar");

    if (!rowEl) return;

    if (isDiagnosing) {
      rowEl.classList.add("row-dimmed");
      if (argonNodeEl) argonNodeEl.classList.add("node-active", "node-pulse");
    } else {
      rowEl.classList.remove("row-dimmed");
      if (argonNodeEl) argonNodeEl.classList.remove("node-active", "node-pulse");
    }

    // Determine active box among the 5 boxes (0..4) strictly from real arrived state
    let activeBoxIndex = -1;
    const newestRecent = prog && Array.isArray(prog.recent) && prog.recent.length > 0 ? prog.recent[0] : null;
    if (isActive && !isDiagnosing) {
      if (isRepairing) {
        activeBoxIndex = 2; // INT8 under test
      } else if (phase === "gate") {
        activeBoxIndex = 4; // Verdict
      } else if (phase === "evaluating" || phase === "re-evaluating") {
        // If a recent evaluated image has arrived, show Compare (3), else INT8 under test (2)
        activeBoxIndex = newestRecent ? 3 : 2;
      }
    } else if (phase === "done" || phase === "needs_human") {
      if (prog && prog.gate) {
        activeBoxIndex = 4;
      }
    }

    const boxes = rowEl.querySelectorAll(".pipeline-node[data-pipe-idx]");
    boxes.forEach((box) => {
      const idx = Number(box.getAttribute("data-pipe-idx"));
      box.classList.remove("node-active", "node-pulse");
      if (idx === activeBoxIndex) {
        box.classList.add("node-active");
        if (isActive) {
          box.classList.add("node-pulse");
        }
      }
    });

    // Repair action chips & thin bar inside INT8 under test box (index 2)
    if (repairChipsEl) {
      repairChipsEl.textContent = "";
      const act = (prog && prog.action) || null;
      if (isRepairing && act) {
        const chips = Array.isArray(act.params_chips) && act.params_chips.length > 0
          ? act.params_chips
          : (act.type ? [act.type] : []);
        chips.forEach((c) => {
          const sp = document.createElement("span");
          sp.className = "pipe-mini-chip";
          sp.textContent = String(c);
          repairChipsEl.appendChild(sp);
        });
      }
    }
    if (repairBarEl) {
      repairBarEl.style.display = isRepairing && isActive ? "block" : "none";
    }

    // Move small photo thumbnail only when a real event arrives from /state
    if (tokenEl && tokenImgEl && tokenLabelEl) {
      const eventItem = newestRecent;
      if (eventItem && (isActive || phase === "done" || phase === "needs_human")) {
        const eventKey = `${phase}:${eventItem.condition || ""}:${eventItem.image_id || ""}:${
          prog && prog.step ? prog.step.done : ""
        }`;
        tokenEl.style.display = "flex";
        if (eventItem.thumb_url) {
          tokenImgEl.src = eventItem.thumb_url;
          tokenImgEl.style.display = "block";
        } else {
          tokenImgEl.style.display = "none";
        }
        tokenLabelEl.textContent = `${eventItem.condition || ""} · ${eventItem.image_id || ""}`;

        const targetHop = activeBoxIndex >= 0 ? activeBoxIndex : 3;
        if (eventKey !== lastPipelineEventKey) {
          lastPipelineEventKey = eventKey;
          currentPipelineHopIndex = targetHop;
        } else {
          currentPipelineHopIndex = targetHop;
        }
        const pct = Math.min(4, Math.max(0, currentPipelineHopIndex)) * 20;
        tokenEl.style.transform = `translateX(${pct * 4.2}%)`;
      } else {
        tokenEl.style.display = "none";
      }
    }
  }

  function renderStepper(state) {
    const phase = String((state.progress && state.progress.phase) || state.phase || "");
    const prog = state.progress && typeof state.progress === "object" ? state.progress : null;
    const stepBadge = document.getElementById("live-step-counter");
    if (stepBadge) {
      if (prog && prog.step && prog.step.label) {
        const d = isFiniteNum(prog.step.done) ? prog.step.done : "N/A";
        const t = isFiniteNum(prog.step.total) ? prog.step.total : "N/A";
        stepBadge.textContent = `${prog.step.label} (${d}/${t})`;
      } else if (ACTIVE_PHASES.has(phase)) {
        stepBadge.textContent = `Phase: ${phase}`;
      } else {
        stepBadge.textContent = `Phase: ${phase}`;
      }
    }

    const activeIdx = STEP_ORDER.indexOf(phase);
    const isDone = phase === "done";
    const isNeedsHuman = phase === "needs_human";

    STEP_ORDER.forEach((stepKey, idx) => {
      const itemEl = document.querySelector(`.stepper-item[data-step-phase="${stepKey}"]`);
      if (!itemEl) return;
      itemEl.classList.remove("step-active", "step-completed");
      if (stepKey === phase && !TERMINAL_PHASES.has(phase)) {
        itemEl.classList.add("step-active");
      } else if (
        (activeIdx > idx) ||
        isDone ||
        (isNeedsHuman && isFiniteNum(observedPhaseDurations[stepKey]))
      ) {
        itemEl.classList.add("step-completed");
      }
    });

    const connectors = document.querySelectorAll(".stepper-connector-fill");
    connectors.forEach((conn, idx) => {
      let filled = false;
      if (isDone) {
        filled = true;
      } else if (activeIdx > idx) {
        filled = true;
      }
      conn.style.width = filled ? "100%" : "0%";
    });

    updateLiveTimerDOM();
  }

  function createSvgOverlayForFilmCard(item) {
    const fp32Boxes = Array.isArray(item.fp32_boxes_geom)
      ? item.fp32_boxes_geom
      : (Array.isArray(item.boxes_fp32) ? item.boxes_fp32 : null);
    const int8Boxes = Array.isArray(item.int8_boxes_geom)
      ? item.int8_boxes_geom
      : (Array.isArray(item.boxes_int8) ? item.boxes_int8 : null);
    const missedBoxes = Array.isArray(item.missed_boxes_geom) ? item.missed_boxes_geom : null;

    if (!fp32Boxes && !int8Boxes && !missedBoxes) {
      return null;
    }

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("class", "film-svg-overlay");
    const w = Array.isArray(item.image_size) && item.image_size[0] ? item.image_size[0] : 640;
    const h = Array.isArray(item.image_size) && item.image_size[1] ? item.image_size[1] : 426;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("preserveAspectRatio", "none");

    function appendRect(boxCoords, clsName, isMissed) {
      const coords = Array.isArray(boxCoords) ? boxCoords : (boxCoords && Array.isArray(boxCoords.box) ? boxCoords.box : null);
      if (!coords || coords.length < 4) return;
      const x1 = Number(coords[0]);
      const y1 = Number(coords[1]);
      const x2 = Number(coords[2]);
      const y2 = Number(coords[3]);
      const rw = Math.max(2, x2 - x1);
      const rh = Math.max(2, y2 - y1);
      const perim = Math.round(2 * (rw + rh));

      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("x", String(x1));
      rect.setAttribute("y", String(y1));
      rect.setAttribute("width", String(rw));
      rect.setAttribute("height", String(rh));
      rect.setAttribute("class", `${clsName}${isMissed ? " svg-box-missed" : ""}`);
      rect.style.strokeDasharray = `${perim}`;
      rect.style.strokeDashoffset = prefersReducedMotion ? "0" : `${perim}`;
      svg.appendChild(rect);
    }

    if (fp32Boxes) {
      fp32Boxes.forEach((b) => {
        const missedFlag = Boolean(b && typeof b === "object" && b.missed);
        appendRect(b, "svg-box-fp32", missedFlag);
      });
    }
    if (missedBoxes) {
      missedBoxes.forEach((b) => appendRect(b, "svg-box-fp32", true));
    }
    if (int8Boxes) {
      int8Boxes.forEach((b) => appendRect(b, "svg-box-int8", false));
    }

    return svg;
  }

  function renderFilmstrip(state) {
    const row = document.getElementById("recent-filmstrip");
    const noteEl = document.getElementById("filmstrip-phase-note");
    if (!row) return;

    const prog = state.progress && typeof state.progress === "object" ? state.progress : null;
    const pPhase = prog ? String(prog.phase || "") : "";

    if (noteEl) {
      if (pPhase === "re-evaluating") {
        noteEl.textContent = "Re-evaluating repaired INT8 model on Split-B photos (boxes redrawn on same photos)";
      } else if (pPhase === "evaluating") {
        noteEl.textContent = "Evaluating baseline INT8 model on Split-B photos";
      } else {
        noteEl.textContent = "Per-image overlays from progress.recent (indoor Split-B photos prioritized first)";
      }
    }

    const rawRecent = prog && Array.isArray(prog.recent) ? prog.recent.slice(0, 8) : [];
    row.textContent = "";

    if (rawRecent.length === 0) {
      const empty = document.createElement("div");
      empty.className = "filmstrip-empty";
      empty.textContent = "Per-image evaluation filmstrip: waiting for evaluated Split-B images.";
      row.appendChild(empty);
      return;
    }

    // Order display so verified indoor objects come first without mutating backend array
    const indexed = rawRecent.map((item, idx) => ({
      item,
      idx,
      indoor: isIndoorPriorityItem(item) ? 0 : 1
    }));
    indexed.sort((a, b) => (a.indoor !== b.indoor ? a.indoor - b.indoor : a.idx - b.idx));

    indexed.forEach(({ item }) => {
      const entryKey = `${pPhase}:${item.condition || ""}:${item.image_id || ""}:${item.int8_boxes ?? ""}`;
      const isNew = !seenRecentKeys.has(entryKey);
      if (isNew) {
        seenRecentKeys.add(entryKey);
      }

      const card = document.createElement("div");
      card.className = `film-card${isNew && !prefersReducedMotion ? " film-enter" : ""}`;
      card.setAttribute("data-image-id", item.image_id || "");
      card.setAttribute("data-condition", item.condition || "");

      const top = document.createElement("div");
      top.className = "film-card-top";
      const condCode = document.createElement("code");
      condCode.textContent = item.condition || "cond";
      const imgCode = document.createElement("code");
      imgCode.textContent = item.image_id || "image";
      top.appendChild(condCode);
      top.appendChild(imgCode);

      const thumbWrap = document.createElement("div");
      thumbWrap.className = "film-thumb-wrap";
      if (item.thumb_url) {
        const img = document.createElement("img");
        img.className = "film-thumb-img";
        img.src = item.thumb_url;
        img.alt = `${item.condition || ""} ${item.image_id || ""}`;
        thumbWrap.appendChild(img);
      } else {
        const na = document.createElement("div");
        na.className = "film-thumb-na";
        na.textContent = "No overlay URL";
        thumbWrap.appendChild(na);
      }

      const svgOverlay = createSvgOverlayForFilmCard(item);
      if (svgOverlay) {
        thumbWrap.appendChild(svgOverlay);
      }

      const counts = document.createElement("div");
      counts.className = "film-counts";
      counts.textContent =
        `FP32 ${fmtInt(item.fp32_boxes)} | INT8 ${fmtInt(item.int8_boxes)} | missed ${fmtInt(item.missed)}`;

      card.appendChild(top);
      card.appendChild(thumbWrap);
      card.appendChild(counts);

      card.addEventListener("click", () => {
        if (item.condition) {
          selectConditionForInspection(item.condition, true, item.image_id);
        }
      });

      row.appendChild(card);
    });
  }

  function renderLiveDiagnosisAndGate(state) {
    const prog = state.progress && typeof state.progress === "object" ? state.progress : null;
    const ar = prog && prog.argon && typeof prog.argon === "object" ? prog.argon : null;

    const sendStep = document.getElementById("argon-step-sending");
    const thinkStep = document.getElementById("argon-step-thinking");
    const recvStep = document.getElementById("argon-step-received");

    if (sendStep) {
      sendStep.className = `argon-step${ar && ar.status === "sending" ? " active" : ""}`;
      const strong = sendStep.querySelector("strong");
      if (strong) strong.textContent = ar && isFiniteNum(ar.images_sent) ? String(ar.images_sent) : "N/A";
    }
    if (thinkStep) {
      thinkStep.className = `argon-step${ar && ar.status === "thinking" ? " active" : ""}`;
    }
    if (recvStep) {
      recvStep.className = `argon-step${ar && ar.status === "received" ? " active" : ""}`;
      const strong = recvStep.querySelector("strong");
      if (strong) strong.textContent = ar && ar.status ? String(ar.status) : "pending";
    }

    const tokensLine = document.getElementById("argon-live-tokens");
    if (tokensLine) {
      let inTok = ar && isFiniteNum(ar.input_tokens) ? ar.input_tokens : null;
      let outTok = ar && isFiniteNum(ar.output_tokens) ? ar.output_tokens : null;
      const logs = Array.isArray(state.argon_logs) ? state.argon_logs : [];
      if (inTok === null && logs.length > 0 && logs[logs.length - 1].usage) {
        const u = logs[logs.length - 1].usage;
        if (isFiniteNum(u.total_input_tokens)) inTok = u.total_input_tokens;
        if (isFiniteNum(u.total_output_tokens)) outTok = u.total_output_tokens;
      }
      tokensLine.textContent =
        `Measured Input Tokens: ${inTok !== null ? inTok : "N/A"} | ` +
        `Measured Output Tokens: ${outTok !== null ? outTok : "N/A"}`;
    }

    // Verbatim diagnosis lines (120ms stagger when new text arrives)
    const diagBox = document.getElementById("live-diagnosis-verbatim");
    if (diagBox) {
      const logs = Array.isArray(state.argon_logs) ? state.argon_logs : [];
      const latestLog = logs.length > 0 ? logs[logs.length - 1] : null;
      const lines = [];
      if (latestLog && Array.isArray(latestLog.diagnosis)) {
        latestLog.diagnosis.forEach((d) => {
          lines.push(`[${d.condition || "condition"} | cause: ${d.cause || "N/A"}] ${d.pattern || ""}`);
        });
      }
      if (latestLog && latestLog.rationale) {
        lines.push(`Rationale: ${latestLog.rationale}`);
      }
      const sig = lines.join("
");
      if (lines.length === 0) {
        diagBox.textContent = "Diagnosis text pending Argon response.";
        lastDiagnosisSignature = "";
      } else if (sig !== lastDiagnosisSignature) {
        lastDiagnosisSignature = sig;
        diagBox.textContent = "";
        lines.forEach((lineText, idx) => {
          const lineDiv = document.createElement("div");
          lineDiv.className = prefersReducedMotion ? "diag-line" : "diag-line diag-line-fade";
          if (!prefersReducedMotion) {
            lineDiv.style.animationDelay = `${idx * 120}ms`;
          }
          lineDiv.textContent = lineText;
          diagBox.appendChild(lineDiv);
        });
      }
    }

    // Action chips
    const actionRow = document.getElementById("live-action-row");
    if (actionRow) {
      actionRow.textContent = "";
      const act = (prog && prog.action) ||
        (Array.isArray(state.argon_logs) && state.argon_logs.length > 0
          ? state.argon_logs[state.argon_logs.length - 1].action
          : null);
      if (act && act.type) {
        const badge = document.createElement("span");
        badge.className = "action-type-badge";
        badge.textContent = `Action: ${act.type}`;
        actionRow.appendChild(badge);
        const chips = Array.isArray(act.params_chips) ? act.params_chips : [];
        chips.forEach((chip) => {
          const sp = document.createElement("span");
          sp.className = "param-chip";
          sp.textContent = String(chip);
          actionRow.appendChild(sp);
        });
      } else {
        const muted = document.createElement("span");
        muted.className = "muted-label";
        muted.textContent = "Repair action: pending diagnosis";
        actionRow.appendChild(muted);
      }
    }

    // Gate banner (slides down 300ms on new verdict and holds)
    const gateWrap = document.getElementById("live-gate-banner-wrap");
    if (gateWrap) {
      const gt = prog && prog.gate && typeof prog.gate === "object" ? prog.gate : null;
      const gateSig = gt ? JSON.stringify(gt) : "";
      if (gateSig !== lastGateSignature) {
        lastGateSignature = gateSig;
        gateWrap.textContent = "";
        if (gt && gt.verdict) {
          const isAcc = String(gt.verdict).toUpperCase().startsWith("ACCEPTED");
          const banner = document.createElement("div");
          banner.className = `gate-banner ${isAcc ? "gate-accepted" : "gate-rejected"}${
            !prefersReducedMotion ? " gate-slide-down" : ""
          }`;
          const title = document.createElement("div");
          title.className = "gate-verdict-title";
          title.textContent = `Gate Verdict: ${gt.verdict}`;
          const details = document.createElement("div");
          details.className = "gate-verdict-details";
          const br = gt.budget_ratios && typeof gt.budget_ratios === "object" ? gt.budget_ratios : {};
          const parts = [
            `Control F1 delta: ${isFiniteNum(gt.control_delta) ? gt.control_delta.toFixed(4) : "N/A"}`,
            `Size vs baseline: ${br.size_ratio_vs_baseline ?? "N/A"}x`,
            `Latency vs baseline: ${br.ms_ratio_vs_baseline ?? "N/A"}x`,
            `Quantized ops: ${br.ops_ratio_vs_baseline ?? "N/A"}x`
          ];
          if (Array.isArray(gt.reasons) && gt.reasons.length > 0) {
            parts.push(`Reasons: ${gt.reasons.join(", ")}`);
          }
          details.textContent = parts.join(" | ");
          banner.appendChild(title);
          banner.appendChild(details);
          gateWrap.appendChild(banner);
        } else {
          const neutral = document.createElement("div");
          neutral.className = "gate-banner gate-neutral";
          neutral.textContent =
            "Gate verdict: pending evaluation against control drop (≤ 0.02), passing conditions, and compression budget.";
          gateWrap.appendChild(neutral);
        }
      }
    }

    // Rejected attempt history (no repeat animation on polls)
    const rejList = document.getElementById("rejected-history-list");
    if (rejList) {
      const iterations = Array.isArray(state.iterations) ? state.iterations : [];
      const logs = Array.isArray(state.argon_logs) ? state.argon_logs : [];
      const rejectedItems = [];

      iterations.forEach((it, idx) => {
        if (!it || typeof it !== "object") return;
        const v = String(it.gate_verdict || (it.gate && it.gate.verdict) || "");
        if (v.toUpperCase().startsWith("REJECTED")) {
          const actType = (it.action && it.action.type) || it.action_type || "repair";
          const reasons = (it.gate && Array.isArray(it.gate.reasons) ? it.gate.reasons.join(", ") : "") || v;
          rejectedItems.push(`Iter ${it.iteration || idx + 1} (${actType}): ${reasons}`);
        }
      });

      if (rejectedItems.length === 0) {
        logs.forEach((lg) => {
          if (lg && String(lg.gate_verdict || "").toUpperCase().startsWith("REJECTED")) {
            const actType = (lg.action && lg.action.type) || "repair";
            rejectedItems.push(`Iter ${lg.iteration} (${actType}): ${lg.gate_verdict}`);
          }
        });
      }

      rejList.textContent = "";
      if (rejectedItems.length === 0) {
        const sp = document.createElement("span");
        sp.className = "muted-label";
        sp.textContent = "No rejected repair attempts recorded in this run.";
        rejList.appendChild(sp);
      } else {
        rejectedItems.forEach((txt) => {
          const rowDiv = document.createElement("div");
          rowDiv.className = "rejected-history-item";
          rowDiv.textContent = txt;
          rejList.appendChild(rowDiv);
        });
      }
    }
  }

  function renderConditionCards(state) {
    const dispThresh = getDisplayPassThreshold(state, null);
    const sanityThresh = getControlSanityThreshold(state);
    const ctrlF1 = state.baseline_control && typeof state.baseline_control === "object" ? state.baseline_control.f1 : null;

    const passEl = document.getElementById("cards-pass-threshold");
    if (passEl) passEl.textContent = fmtF1(dispThresh);
    const ctrlEl = document.getElementById("cards-control-f1");
    if (ctrlEl) ctrlEl.textContent = fmtF1(ctrlF1);
    const sanEl = document.getElementById("cards-sanity-threshold");
    if (sanEl) sanEl.textContent = fmtNum(sanityThresh, 2);

    const conditionsMap = state.conditions && typeof state.conditions === "object" ? state.conditions : {};
    const manifestMap = state.manifest_summary && typeof state.manifest_summary === "object" ? state.manifest_summary : {};
    const profileConds =
      state.scenario_meta && Array.isArray(state.scenario_meta.conditions) ? state.scenario_meta.conditions : [];

    profileConds.forEach((condCfg) => {
      const cid = condCfg.id;
      const cardEl = document.querySelector(`.cond-card[data-cond-id="${cid}"]`);
      if (!cardEl) return;

      const cstate = conditionsMap[cid] || null;
      const msum = manifestMap[cid] || null;
      const cm = extractConditionMetrics(cstate, state);
      const cardThresh = getDisplayPassThreshold(state, cstate);

      const wasPass = previousCardPass[cid];
      const flippedToPass = wasPass === false && cm.hasMeasured && cm.isPass;
      previousCardPass[cid] = cm.hasMeasured ? cm.isPass : null;

      cardEl.classList.remove("card-pass", "card-fail", "card-pending", "card-limitation", "card-pass-transition");
      if (cm.hasMeasured) {
        // Block C 14:08 rule: compression pass/fail applies to EVERY condition
        cardEl.classList.add(cm.isPass ? "card-pass" : "card-fail");
        if (flippedToPass && !prefersReducedMotion) {
          cardEl.classList.add("card-pass-transition");
        }
      } else {
        cardEl.classList.add("card-pending");
      }

      // Badges: show PASS / FAIL and informational model limitation tag alongside if applicable
      const badgesWrap = cardEl.querySelector(".cond-badges");
      if (badgesWrap) {
        badgesWrap.textContent = "";
        if (cm.hasMeasured) {
          const verdictBadge = document.createElement("span");
          verdictBadge.className = `badge ${cm.isPass ? "badge-pass" : "badge-fail"}`;
          verdictBadge.textContent = cm.isPass ? "PASS" : "FAIL";
          badgesWrap.appendChild(verdictBadge);

          if (cm.isLimitation) {
            const limBadge = document.createElement("span");
            limBadge.className = "badge badge-limitation";
            limBadge.textContent = "model limitation: FP32 also degrades";
            badgesWrap.appendChild(limBadge);
          }
        } else {
          const pendBadge = document.createElement("span");
          pendBadge.className = "badge badge-pending";
          pendBadge.textContent = "evaluation pending";
          badgesWrap.appendChild(pendBadge);
        }
      }

      // Update card body (Hero F1 + Object counts OR pending thumbnail)
      const bodySlot = cardEl.querySelector(".card-dynamic-slot");
      if (bodySlot) {
        if (cm.hasMeasured) {
          let heroWrap = bodySlot.querySelector(".card-hero-metrics");
          if (!heroWrap) {
            bodySlot.textContent = "";
            heroWrap = document.createElement("div");
            heroWrap.className = "card-hero-metrics";
            const f1Div = document.createElement("div");
            f1Div.className = "hero-f1-value";
            const objDiv = document.createElement("div");
            objDiv.className = "hero-object-counts";
            const mainObj = document.createElement("div");
            mainObj.className = "obj-count-main";
            const subObj = document.createElement("div");
            subObj.className = "obj-count-sub";
            objDiv.appendChild(mainObj);
            objDiv.appendChild(subObj);
            heroWrap.appendChild(f1Div);
            heroWrap.appendChild(objDiv);
            bodySlot.appendChild(heroWrap);
          }

          const f1Div = heroWrap.querySelector(".hero-f1-value");
          if (f1Div) {
            f1Div.className = `hero-f1-value ${cm.isPass ? "text-pass" : "text-fail"}`;
            const prevVal = previousCardF1[cid];
            const curTarget = isFiniteNum(cm.finalF1) ? cm.finalF1 : cm.baselineF1;
            if (isFiniteNum(prevVal) && isFiniteNum(curTarget) && Math.abs(prevVal - curTarget) > 1e-6) {
              const prefix = isFiniteNum(cm.finalF1) ? `${fmtF1(cm.baselineF1)} → ` : "";
              const suffix = isFiniteNum(cm.finalF1) ? "" : " → N/A";
              animateValueText(f1Div, prevVal, curTarget, 600, prefix, suffix);
            } else {
              f1Div.textContent = `${fmtF1(cm.baselineF1)} → ${fmtF1(cm.finalF1)}`;
            }
            if (isFiniteNum(curTarget)) {
              previousCardF1[cid] = curTarget;
            }
          }

          const mainObj = heroWrap.querySelector(".obj-count-main");
          if (mainObj) {
            const imgScope = isFiniteNum(cm.validImages) && isFiniteNum(cm.nImages)
              ? ` (${cm.validImages}/${cm.nImages} valid Split-B imgs)`
              : (isFiniteNum(cm.nImages) ? ` (${cm.nImages} Split-B imgs)` : "");
            mainObj.textContent = `|R| = ${fmtInt(cm.refObjects)} ref objects${imgScope}`;
          }
          const subObj = heroWrap.querySelector(".obj-count-sub");
          if (subObj) {
            subObj.textContent = `TP ${fmtInt(cm.tp)} | FP ${fmtInt(cm.fp)} | FN ${fmtInt(cm.fn)}`;
          }
        }
      }

      // Threshold line
      const threshLine = cardEl.querySelector(".card-threshold-line");
      if (threshLine) {
        threshLine.textContent =
          `Pass threshold: ≥ ${fmtF1(cardThresh)} (Control F1 ${fmtF1(ctrlF1)} − 0.05) | ` +
          `Control sanity: ≥ ${fmtNum(sanityThresh, 2)}`;
      }

      // Provenance / Generator summary from actual manifest (never claim Nano Banana succeeded when fallback ran)
      const provEl = cardEl.querySelector(".cond-provenance-text");
      if (provEl) {
        const actualGen =
          (cstate && cstate.generator_summary) ||
          (msum && msum.generator_label) ||
          (msum && Array.isArray(msum.generators_used) && msum.generators_used.length > 0
            ? msum.generators_used.join(", ")
            : "pending manifest");
        const parts = [`Actual generator (Split B): ${actualGen}`];
        if (msum && isFiniteNum(msum.valid_b_count) && isFiniteNum(msum.split_b_count)) {
          parts.push(`Valid Split B: ${msum.valid_b_count}/${msum.split_b_count}`);
        }
        if (msum && isFiniteNum(msum.fallback_b_count) && msum.fallback_b_count > 0) {
          parts.push(`Fallback used: ${msum.fallback_b_count}${msum.fallback_reason ? " (" + msum.fallback_reason + ")" : ""}`);
        }
        if (isFiniteNum(cm.retention)) {
          parts.push(`FP32 retention: ${cm.retention.toFixed(2)}`);
        } else if (msum && isFiniteNum(msum.retention_b)) {
          parts.push(`FP32 retention: ${msum.retention_b.toFixed(2)}`);
        }
        provEl.textContent = parts.join(" | ");
      }
    });

    // Auto-select first failing condition (e.g., fog) if user hasn't picked one manually
    if (!userPickedInspectCondition) {
      let firstFailId = null;
      for (const condCfg of profileConds) {
        const cm = extractConditionMetrics(conditionsMap[condCfg.id], state);
        if (cm.hasMeasured && !cm.isPass) {
          firstFailId = condCfg.id;
          break;
        }
      }
      const targetId = firstFailId || selectedInspectCondition || (profileConds[0] && profileConds[0].id) || "fog";
      if (targetId !== selectedInspectCondition) {
        selectedInspectCondition = targetId;
      }
    }
  }

  function resolveEvidenceImageForCondition(state, condId, explicitImgId) {
    const msum = state.manifest_summary && state.manifest_summary[condId] ? state.manifest_summary[condId] : null;
    const byId = msum && msum.entries_by_id && typeof msum.entries_by_id === "object" ? msum.entries_by_id : {};
    const cwFile = (state.crash_wall && (state.crash_wall.file_name || state.crash_wall.image_id)) || "000000000139.jpg";

    if (explicitImgId) {
      const stem = normStem(explicitImgId);
      if (byId[explicitImgId]) return byId[explicitImgId].file_name || explicitImgId;
      if (byId[stem]) return byId[stem].file_name || explicitImgId;
      return explicitImgId.includes(".") ? explicitImgId : `${explicitImgId}.jpg`;
    }
    return cwFile;
  }

  function renderExpandedComparison(state, overrideImageId) {
    const condId = selectedInspectCondition || "fog";
    const scenarioId = state.scenario_id || currentScenario;
    const conditionsMap = state.conditions && typeof state.conditions === "object" ? state.conditions : {};
    const cstate = conditionsMap[condId] || null;
    const cm = extractConditionMetrics(cstate, state);

    // Highlight selector button
    const selBtns = document.querySelectorAll(".cond-select-btn");
    selBtns.forEach((btn) => {
      if (btn.getAttribute("data-select-cond") === condId) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    const imgFile = resolveEvidenceImageForCondition(state, condId, overrideImageId);
    const comp = cstate && cstate.comparison && typeof cstate.comparison === "object" ? cstate.comparison : null;

    // Panel 1: Original + FP32 reference
    const p1Id = document.getElementById("panel-1-img-id");
    const p1Body = document.getElementById("panel-1-body");
    if (p1Id) p1Id.textContent = imgFile || "N/A";
    if (p1Body) {
      p1Body.textContent = "";
      const origUrl = (comp && comp.original_url) || (imgFile ? `/preview/${scenarioId}/original/${imgFile}?model=fp32` : null);
      if (origUrl) {
        const img = document.createElement("img");
        img.className = "panel-img";
        img.src = origUrl;
        img.alt = `Original Split-B (${imgFile}) + FP32 Reference`;
        p1Body.appendChild(img);
      } else {
        const na = document.createElement("div");
        na.className = "panel-na-box";
        na.textContent = "N/A — Original Split-B preview not available";
        p1Body.appendChild(na);
      }
    }

    // Panel 2: Condition + Baseline INT8 overlay
    const p2Id = document.getElementById("panel-2-img-id");
    const p2Body = document.getElementById("panel-2-body");
    if (p2Id) p2Id.textContent = `${condId} · ${imgFile || "N/A"}`;
    if (p2Body) {
      p2Body.textContent = "";
      const baseUrl =
        (comp && comp.baseline_url) || (imgFile ? `/preview/${scenarioId}/${condId}/${imgFile}?model=baseline` : null);
      if (baseUrl) {
        const img = document.createElement("img");
        img.className = "panel-img";
        img.src = baseUrl;
        img.alt = `${condId} + Baseline INT8 Overlay`;
        p2Body.appendChild(img);
      } else {
        const na = document.createElement("div");
        na.className = "panel-na-box";
        na.textContent = "N/A — Baseline condition overlay not yet available";
        p2Body.appendChild(na);
      }
    }

    // Panel 3: Condition + Repaired INT8 overlay (NEVER substitute baseline overlay if repaired is missing)
    const p3Id = document.getElementById("panel-3-img-id");
    const p3Body = document.getElementById("panel-3-body");
    if (p3Id) p3Id.textContent = `${condId} · ${imgFile || "N/A"}`;
    if (p3Body) {
      p3Body.textContent = "";
      const repUrl =
        (comp && comp.repaired_url) ||
        (state.has_repaired_model && imgFile ? `/preview/${scenarioId}/${condId}/${imgFile}?model=repaired` : null);
      if (repUrl) {
        const img = document.createElement("img");
        img.className = "panel-img";
        img.src = repUrl;
        img.alt = `${condId} + Repaired INT8 Overlay`;
        p3Body.appendChild(img);
      } else {
        const na = document.createElement("div");
        na.className = "panel-na-box";
        na.textContent = "N/A — Repaired INT8 overlay not yet available (pending repair)";
        p3Body.appendChild(na);
      }
    }

    // Panel 4: Per-class F1 & Per-size recall deltas
    const p4Label = document.getElementById("panel-4-cond-label");
    if (p4Label) {
      p4Label.textContent = `Condition: ${condId} (${cm.hasMeasured ? cm.statusLabel : "pending"})`;
    }
    const tbody = document.getElementById("panel-4-table-body");
    if (tbody) {
      tbody.textContent = "";
      const bm = (cstate && cstate.baseline_metrics) || cstate || {};
      const fm = (cstate && cstate.final_metrics) || null;
      const bSize = (comp && comp.per_size_baseline) || (bm && bm.per_size) || {};
      const fSize = (comp && comp.per_size_final) || (fm && fm.per_size) || {};
      const bClass = (comp && comp.per_class_baseline) || (bm && bm.per_class) || {};
      const fClass = (comp && comp.per_class_final) || (fm && fm.per_class) || {};
      const clsNames = (state.scenario_meta && state.scenario_meta.class_names) || {};

      function addDeltaRow(label, bVal, fVal) {
        const tr = document.createElement("tr");
        const tdLabel = document.createElement("td");
        tdLabel.textContent = label;
        const tdBase = document.createElement("td");
        tdBase.textContent = isFiniteNum(bVal) ? bVal.toFixed(4) : "N/A";
        const tdFinal = document.createElement("td");
        tdFinal.textContent = isFiniteNum(fVal) ? fVal.toFixed(4) : "N/A";
        const tdDelta = document.createElement("td");
        if (isFiniteNum(bVal) && isFiniteNum(fVal)) {
          const d = fVal - bVal;
          tdDelta.textContent = `${d >= 0 ? "+" : ""}${d.toFixed(4)}`;
          tdDelta.className = d > 0 ? "text-pass" : (d < 0 ? "text-fail" : "");
        } else {
          tdDelta.textContent = "N/A";
        }
        tr.appendChild(tdLabel);
        tr.appendChild(tdBase);
        tr.appendChild(tdFinal);
        tr.appendChild(tdDelta);
        tbody.appendChild(tr);
      }

      addDeltaRow("Overall F1", cm.baselineF1, cm.finalF1);
      ["small", "medium", "large"].forEach((sz) => {
        const bv = isFiniteNum(bSize[sz]) ? bSize[sz] : null;
        const fv = isFiniteNum(fSize[sz]) ? fSize[sz] : null;
        addDeltaRow(`Size Recall: ${sz}`, bv, fv);
      });

      const allClassKeys = Array.from(new Set([...Object.keys(bClass), ...Object.keys(fClass)]));
      allClassKeys.forEach((cidKey) => {
        const cname = clsNames[cidKey] ? `${clsNames[cidKey]} (${cidKey})` : `Class ${cidKey}`;
        const bv = isFiniteNum(bClass[cidKey]) ? bClass[cidKey] : null;
        const fv = isFiniteNum(fClass[cidKey]) ? fClass[cidKey] : null;
        addDeltaRow(`Class F1: ${cname}`, bv, fv);
      });
    }

    // Expanded verbatim diagnosis + cited evidence images
    const diagPre = document.getElementById("expanded-diagnosis-text");
    const claimsListEl = document.getElementById("expanded-claims-list");
    const citedGrid = document.getElementById("cited-evidence-grid");

    const logs = Array.isArray(state.argon_logs) ? state.argon_logs : [];
    let matchedDiagEntries = [];
    let matchedRationale = "";
    logs.forEach((lg) => {
      if (lg && Array.isArray(lg.diagnosis)) {
        lg.diagnosis.forEach((d) => {
          if (!d.condition || d.condition === condId) {
            matchedDiagEntries.push(d);
            if (lg.rationale) matchedRationale = lg.rationale;
          }
        });
      }
    });

    if (diagPre) {
      const NL = String.fromCharCode(10);
      if (matchedDiagEntries.length > 0) {
        const txtLines = matchedDiagEntries.map(
          (d) =>
            [
              `Condition: ${d.condition || condId}`,
              `Cause: ${d.cause || "N/A"}`,
              `Pattern: ${d.pattern || "N/A"}`,
              `Evidence IDs: ${Array.isArray(d.evidence_images) ? d.evidence_images.join(", ") : "N/A"}`
            ].join(NL)
        );
        if (matchedRationale) {
          txtLines.push(`${NL}Rationale: ${matchedRationale}`);
        }
        diagPre.textContent = txtLines.join(NL + NL);
      } else {
        diagPre.textContent = `N/A — Argon diagnosis has not been recorded for condition '${condId}' yet.`;
      }
    }

    if (claimsListEl) {
      claimsListEl.textContent = "";
      matchedDiagEntries.forEach((d) => {
        if (Array.isArray(d.claims)) {
          d.claims.forEach((cl) => {
            const rowDiv = document.createElement("div");
            rowDiv.className = "claim-item";
            const left = document.createElement("span");
            left.textContent = `Claim: ${cl.condition || condId} · ${cl.field || "metric"} = ${cl.value ?? "N/A"}`;
            const right = document.createElement("span");
            const ver = cl.verified === true ? "verified" : (cl.verified === false ? "mismatch" : "unverified");
            right.className = `badge ${ver === "verified" ? "badge-pass" : (ver === "mismatch" ? "badge-fail" : "badge-pending")}`;
            right.textContent = ver;
            rowDiv.appendChild(left);
            rowDiv.appendChild(right);
            claimsListEl.appendChild(rowDiv);
          });
        }
      });
    }

    if (citedGrid) {
      citedGrid.textContent = "";
      const citedIds = [];
      matchedDiagEntries.forEach((d) => {
        if (Array.isArray(d.evidence_images)) {
          d.evidence_images.forEach((eid) => {
            if (eid && !citedIds.includes(eid)) citedIds.push(eid);
          });
        }
      });
      if (cstate && Array.isArray(cstate.evidence_images)) {
        cstate.evidence_images.forEach((eid) => {
          if (eid && !citedIds.includes(eid)) citedIds.push(eid);
        });
      }

      if (citedIds.length === 0) {
        const na = document.createElement("div");
        na.className = "panel-na-box";
        na.textContent = "No cited evidence images recorded for this condition yet.";
        citedGrid.appendChild(na);
      } else {
        const recentArr = state.progress && Array.isArray(state.progress.recent) ? state.progress.recent : [];
        citedIds.forEach((rawId) => {
          const resolvedFile = resolveEvidenceImageForCondition(state, condId, rawId);
          const stem = normStem(resolvedFile);
          const recMatch = recentArr.find(
            (r) =>
              r.condition === condId &&
              (normStem(r.image_id) === stem || normStem(r.image_id) === normStem(rawId))
          );
          const imgUrl =
            (recMatch && recMatch.thumb_url) || `/preview/${scenarioId}/${condId}/${resolvedFile}?model=baseline`;

          const card = document.createElement("div");
          card.className = `evidence-enlarged-card${!prefersReducedMotion ? " evidence-scale-in" : ""}`;

          const hdr = document.createElement("div");
          hdr.className = "evidence-card-header";
          const s1 = document.createElement("span");
          s1.textContent = `Cited: ${rawId}`;
          const s2 = document.createElement("span");
          s2.textContent = resolvedFile;
          hdr.appendChild(s1);
          hdr.appendChild(s2);

          const img = document.createElement("img");
          img.className = "evidence-card-img";
          img.src = imgUrl;
          img.alt = `Cited evidence ${rawId}`;

          const ftr = document.createElement("div");
          ftr.className = "evidence-card-footer";
          ftr.textContent = recMatch
            ? `FP32 ${fmtInt(recMatch.fp32_boxes)} | INT8 ${fmtInt(recMatch.int8_boxes)} | missed ${fmtInt(recMatch.missed)}`
            : `Condition: ${condId} (${resolvedFile})`;

          card.appendChild(hdr);
          card.appendChild(img);
          card.appendChild(ftr);
          citedGrid.appendChild(card);
        });
      }
    }
  }

  function renderArgonLogTable(state) {
    const container = document.getElementById("argon-log-container");
    if (!container) return;
    const logs = Array.isArray(state.argon_logs) ? state.argon_logs : [];
    container.textContent = "";

    if (logs.length === 0) {
      const p = document.createElement("p");
      p.className = "empty-notice";
      p.textContent = "Argon diagnosis and repair loop: not yet run";
      container.appendChild(p);
      return;
    }

    const table = document.createElement("table");
    table.className = "data-table";
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    [
      "Iter",
      "Interaction ID",
      "Thinking",
      "Input Tokens",
      "Output Tokens",
      "Thought Tokens",
      "Latency (s)",
      "Action",
      "Gate Verdict",
      "Cache / Review"
    ].forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    logs.forEach((log) => {
      const tr = document.createElement("tr");
      const u = log.usage || {};
      const actType = (log.action && log.action.type) || "N/A";
      const cacheLabel = log.cached ? "cached from identical config" : "measured";
      const cells = [
        String(log.iteration ?? "N/A"),
        String(log.interaction_id || "N/A"),
        String(log.thinking_level || "high"),
        isFiniteNum(u.total_input_tokens) ? String(u.total_input_tokens) : "N/A",
        isFiniteNum(u.total_output_tokens) ? String(u.total_output_tokens) : "N/A",
        isFiniteNum(u.total_thought_tokens) ? String(u.total_thought_tokens) : "N/A",
        isFiniteNum(log.latency_s) ? log.latency_s.toFixed(2) : "N/A",
        actType,
        String(log.gate_verdict || "N/A"),
        `${cacheLabel}${log.needs_human ? " | needs_human" : ""}`
      ];
      cells.forEach((val, idx) => {
        const td = document.createElement("td");
        if (idx === 1 || idx === 7) {
          const code = document.createElement("code");
          code.textContent = val;
          td.appendChild(code);
        } else if (idx === 8) {
          const st = document.createElement("strong");
          st.textContent = val;
          td.appendChild(st);
        } else {
          td.textContent = val;
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  }

  function applyState(state) {
    if (!state || typeof state !== "object") return;
    currentState = state;
    syncTimersFromState(state);
    renderTopBannersAndStatus(state);
    renderCrashWall(state);
    renderSummaryStrip(state);
    renderPipelineSchematic(state);
    renderStepper(state);
    renderFilmstrip(state);
    renderLiveDiagnosisAndGate(state);
    renderConditionCards(state);
    renderExpandedComparison(state, null);
    renderArgonLogTable(state);
  }

  function selectConditionForInspection(condId, manualClick, overrideImageId) {
    selectedInspectCondition = condId;
    if (manualClick) {
      userPickedInspectCondition = true;
    }
    if (currentState) {
      renderExpandedComparison(currentState, overrideImageId || null);
    }
  }

  async function pollState() {
    if (isMockMode || isReplayMode) return;
    try {
      const qs = new URLSearchParams();
      qs.set("scenario", currentScenario);
      if (activeRunId) {
        qs.set("run_id", activeRunId);
      }
      const resp = await fetch(`/state?${qs.toString()}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data && data.run_id && data.run_id !== "showcase_default") {
        activeRunId = data.run_id;
      }
      applyState(data);
    } catch (err) {
      console.error("State poll error:", err);
    }
  }

  async function triggerLiveConditionRun(condId) {
    selectedLiveCondition = condId;
    if (isMockMode) {
      startMockSequencePlayback(condId);
      return;
    }

    try {
      const resp = await fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario_id: currentScenario,
          conditions: [condId],
          n_images: 20,
          mode: "live"
        })
      });
      if (resp.status === 409) {
        await pollState();
        return;
      }
      const data = await resp.json();
      if (data && data.run_id) {
        activeRunId = data.run_id;
        seenRecentKeys.clear();
        Object.keys(observedPhaseDurations).forEach((k) => delete observedPhaseDurations[k]);
        await pollState();
      }
    } catch (err) {
      console.error("Failed to trigger live run:", err);
    }
  }

  async function submitHumanDecision(decision) {
    if (isMockMode) {
      if (currentState) {
        currentState.human_decision = decision;
        applyState(currentState);
      }
      return;
    }
    try {
      const resp = await fetch("/human_decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: activeRunId,
          decision
        })
      });
      if (resp.ok) {
        await pollState();
      }
    } catch (err) {
      console.error("Failed to record human decision:", err);
    }
  }

  async function initMockMode() {
    try {
      const resp = await fetch("/static/mock_state.json");
      if (!resp.ok) return;
      const mockPayload = await resp.json();
      if (Array.isArray(mockPayload.sequence) && mockPayload.sequence.length > 0) {
        mockSequence = mockPayload.sequence;
        const firstFrame = mergeMockFrameWithBase(mockSequence[0]);
        applyState(firstFrame);
      } else if (mockPayload && typeof mockPayload === "object") {
        applyState(mergeMockFrameWithBase(mockPayload));
      }
    } catch (err) {
      console.error("Failed to load mock_state.json:", err);
    }
  }

  function mergeMockFrameWithBase(frame) {
    const base = currentState ? JSON.parse(JSON.stringify(currentState)) : {};
    const merged = Object.assign({}, base, JSON.parse(JSON.stringify(frame)));
    if (frame.conditions && base.conditions) {
      merged.conditions = Object.assign({}, base.conditions, JSON.parse(JSON.stringify(frame.conditions)));
    }
    merged.is_mock_preview = true;
    return merged;
  }

  function startMockSequencePlayback(condId) {
    if (mockIntervalId) {
      clearInterval(mockIntervalId);
      mockIntervalId = null;
    }
    if (!Array.isArray(mockSequence) || mockSequence.length === 0) return;
    seenRecentKeys.clear();
    Object.keys(observedPhaseDurations).forEach((k) => delete observedPhaseDurations[k]);
    mockStepIndex = 0;

    function applyNextMockStep() {
      if (mockStepIndex >= mockSequence.length) {
        clearInterval(mockIntervalId);
        mockIntervalId = null;
        return;
      }
      const frame = mergeMockFrameWithBase(mockSequence[mockStepIndex]);
      if (frame.progress && frame.progress.phase) {
        frame.progress.phase_started_at = new Date(Date.now() - (frame.progress.phase_elapsed_s || 0) * 1000).toISOString();
      }
      if (condId && frame.progress && Array.isArray(frame.progress.recent)) {
        frame.progress.recent.forEach((r) => {
          r.condition = condId;
        });
      }
      applyState(frame);
      mockStepIndex += 1;
    }

    applyNextMockStep();
    mockIntervalId = setInterval(applyNextMockStep, 1400);
  }

  async function initReplayMode() {
    try {
      const resp = await fetch(`/replay_events?scenario=${encodeURIComponent(currentScenario)}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (!data.available || !Array.isArray(data.events) || data.events.length === 0) return;
      let idx = 0;
      const timer = setInterval(() => {
        if (idx >= data.events.length) {
          clearInterval(timer);
          return;
        }
        const ev = mergeMockFrameWithBase(data.events[idx]);
        applyState(ev);
        idx += 1;
      }, 500);
    } catch (err) {
      console.error("Replay error:", err);
    }
  }

  function bindEvents() {
    // 3 large condition run buttons (Night / Glare / Fog)
    const condBtns = document.querySelectorAll(".run-cond-btn");
    condBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const cid = btn.getAttribute("data-live-cond") || "night";
        triggerLiveConditionRun(cid);
      });
    });

    // Legacy fallback button if present
    const legacyRunBtn = document.getElementById("run-live-btn");
    if (legacyRunBtn) {
      legacyRunBtn.addEventListener("click", () => {
        triggerLiveConditionRun(selectedLiveCondition || "night");
      });
    }

    // Condition card inspect buttons & expanded selector buttons
    document.addEventListener("click", (ev) => {
      const t = ev.target;
      if (!(t instanceof HTMLElement)) return;
      const inspectCond = t.getAttribute("data-inspect-cond");
      if (inspectCond) {
        selectConditionForInspection(inspectCond, true, null);
        const sec = document.getElementById("expanded-comparison-section");
        if (sec && sec.scrollIntoView) {
          sec.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
        }
        return;
      }
      const selectCond = t.getAttribute("data-select-cond");
      if (selectCond) {
        selectConditionForInspection(selectCond, true, null);
      }
    });
  }

  // Bootstrap from embedded initial state JSON
  const initialScript = document.getElementById("initial-state-data");
  if (initialScript && initialScript.textContent) {
    try {
      const parsed = JSON.parse(initialScript.textContent);
      applyState(parsed);
    } catch (err) {
      console.error("Failed to parse initial state JSON:", err);
    }
  }

  bindEvents();

  // 250ms client tick ONLY advances live phase/argon counters when anchored to real backend timestamps
  setInterval(updateLiveTimerDOM, 250);

  if (isMockMode) {
    initMockMode();
  } else if (isReplayMode) {
    initReplayMode();
  } else {
    setInterval(pollState, 2000);
  }
})();
