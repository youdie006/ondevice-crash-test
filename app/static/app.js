(function () {
  const runBtn = document.getElementById("run-live-btn");
  const statusBadge = document.getElementById("run-status-badge");
  let activeRunId = null;

  async function pollState() {
    try {
      const url = activeRunId ? `/state?run_id=${encodeURIComponent(activeRunId)}` : "/state";
      const resp = await fetch(url);
      if (!resp.ok) return;
      const data = await resp.json();
      if (statusBadge && data.phase) {
        statusBadge.textContent = `Phase: ${data.phase}`;
      }
    } catch (err) {
      console.error("State poll error:", err);
    }
  }

  if (runBtn) {
    runBtn.addEventListener("click", async function () {
      runBtn.disabled = true;
      try {
        const resp = await fetch("/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scenario_id: "robot_vacuum",
            conditions: ["night"],
            n_images: 20,
            mode: "live"
          })
        });
        const data = await resp.json();
        if (data.run_id) {
          activeRunId = data.run_id;
          await pollState();
        }
      } catch (err) {
        console.error("Failed to trigger run:", err);
      } finally {
        runBtn.disabled = false;
      }
    });
  }

  setInterval(pollState, 2000);
})();
