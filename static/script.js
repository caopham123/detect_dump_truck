document.addEventListener('DOMContentLoaded', () => {
  const cameraContainer = document.getElementById('camera-container');
  const logContainer = document.getElementById('log-container');
  const totalDumpsEl = document.getElementById('total-dumps');
  const refreshLogsBtn = document.getElementById('refresh-logs');

  // Fetch and setup cameras
  async function setupCameras() {
    try {
      const response = await fetch('/api/cameras');
      const cameras = await response.json();

      // If empty API response, keep existing cards (could be a temporary backend issue), don't clear the UI
      if (!Array.isArray(cameras) || cameras.length === 0) {
        return;
      }

      // When API response returns an empty array but there are existing cards, keep them (backend might be temporarily down).
      const loadingEl = cameraContainer.querySelector('.loading-cameras');
      if (loadingEl && cameraContainer.querySelectorAll('[data-cam-id]').length === 0) {
        cameraContainer.innerHTML = '';
      }

      let totalDumps = 0;

      // Don't remove old cards — always keep enough camera slots even if offline
      const currentCamIds = cameras.map(c => c.id);
      Array.from(cameraContainer.querySelectorAll('[data-cam-id]')).forEach(child => {
        if (!currentCamIds.includes(child.dataset.camId)) {
          child.remove(); // Camera bị xóa khỏi config.yaml thì mới xóa card
        }
      });

      cameras.forEach(cam => {
        totalDumps += cam.dumps;
        const isOnline = cam.status === 'active';

        let camCard = cameraContainer.querySelector(`[data-cam-id="${cam.id}"]`);

        if (!camCard) {
          camCard = document.createElement('div');
          camCard.className = 'camera-card';
          camCard.dataset.camId = cam.id;
          camCard.innerHTML = `
            <div class="camera-header">
              <div class="camera-title">
                <i class="fa-solid fa-video"></i> ${cam.id}
              </div>
              <div class="camera-status" id="status-${cam.id}">Offline</div>
            </div>
            <div class="camera-feed">
              <img src="/video_feed/${cam.id}" alt="Feed ${cam.id}">
              <div class="camera-overlay">
                <span class="cam-badge"><i class="fa-solid fa-layer-group"></i> Default ROI</span>
                <span class="cam-badge cam-dumps" id="dumps-${cam.id}"><i class="fa-solid fa-truck"></i> ${cam.dumps} Dumps</span>
              </div>
            </div>
          `;
          cameraContainer.appendChild(camCard);
        }

        // Update status each time we poll (for both new and existing cards)
        const statusBadge = camCard.querySelector(`#status-${cam.id}`);
        if (statusBadge) {
          statusBadge.textContent = isOnline ? 'Active' : 'Offline';
          statusBadge.className = isOnline ? 'camera-status' : 'camera-status offline';
        }
        camCard.classList.toggle('offline', !isOnline);

        const dumpBadge = camCard.querySelector(`#dumps-${cam.id}`);
        if (dumpBadge) {
          dumpBadge.innerHTML = `<i class="fa-solid fa-truck"></i> ${cam.dumps} Dumps`;
        }
      });

      totalDumpsEl.textContent = totalDumps;
    } catch (error) {
      // Interrupted connection — don't clear the UI, just show error in console and retry on next interval
      console.error('Camera fetch failed, retrying...', error);

    }
  }

  // Fetch logs
  async function fetchLogs() {
    try {
      const response = await fetch('/api/logs');
      const data = await response.json();

      if (!data.logs || data.logs.length === 0) {
        logContainer.innerHTML = '<div class="loading-logs"><p>No logs available for today.</p></div>';
        return;
      }

      logContainer.innerHTML = '';

      data.logs.forEach(logLine => {
        // Parse log line: "2026-06-09 09:15:52 | [INFO] | Message..."
        const match = logLine.match(/^(.*?)\s\|\s\[(.*?)\]\s\|\s(.*)$/);

        const entryDiv = document.createElement('div');

        if (match) {
          const [_, time, level, msg] = match;
          entryDiv.className = `log-entry ${level.toLowerCase()}`;

          // Extract time component from datetime string
          const timeOnly = time.split(' ')[1] || time;

          entryDiv.innerHTML = `
            <span class="log-time">${timeOnly}</span>
            <span class="log-level">[${level}]</span>
            <span class="log-msg">${msg}</span>
          `;
        } else {
          entryDiv.className = 'log-entry';
          entryDiv.textContent = logLine;
        }

        logContainer.appendChild(entryDiv);
      });
    } catch (error) {
      console.error('Error fetching logs:', error);
    }
  }

  // Initialize
  setupCameras();
  fetchLogs();

  // Event listeners
  refreshLogsBtn.addEventListener('click', () => {
    refreshLogsBtn.querySelector('i').classList.add('fa-spin');
    fetchLogs().then(() => {
      setTimeout(() => {
        refreshLogsBtn.querySelector('i').classList.remove('fa-spin');
      }, 500);
    });
  });

  // Auto refresh stats and logs every 5 seconds
  setInterval(() => {
    setupCameras(); // Updates dump counts and active cameras
    fetchLogs();
  }, 5000);
});
