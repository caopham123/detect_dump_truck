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

      if (cameras.length === 0) {
        cameraContainer.innerHTML = '<div class="loading-cameras"><p>No cameras configured or active.</p></div>';
        return;
      }

      // Remove loading indicator if present
      const loadingEl = cameraContainer.querySelector('.loading-cameras');
      if (loadingEl) {
        cameraContainer.innerHTML = '';
      }

      let totalDumps = 0;

      // Track existing cameras to remove ones that disappeared
      const currentCamIds = cameras.map(c => c.id);
      Array.from(cameraContainer.children).forEach(child => {
        if (child.dataset.camId && !currentCamIds.includes(child.dataset.camId)) {
          child.remove();
        }
      });

      cameras.forEach(cam => {
        totalDumps += cam.dumps;

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
              <div class="camera-status">Active</div>
            </div>
            <div class="camera-feed">
              <img src="/video_feed/${cam.id}" alt="Feed ${cam.id}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMzMzMiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZmlsbD0iI2ZmZiIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+U2lnbmFsIExvc3Q8L3RleHQ+PC9zdmc+'">
              <div class="camera-overlay">
                <span class="cam-badge"><i class="fa-solid fa-layer-group"></i> Default ROI</span>
                <span class="cam-badge cam-dumps" id="dumps-${cam.id}"><i class="fa-solid fa-truck"></i> ${cam.dumps} Dumps</span>
              </div>
            </div>
          `;
          cameraContainer.appendChild(camCard);
        } else {
          // Just update the dynamic values without recreating DOM
          const dumpBadge = camCard.querySelector(`#dumps-${cam.id}`);
          if (dumpBadge) {
            dumpBadge.innerHTML = `<i class="fa-solid fa-truck"></i> ${cam.dumps} Dumps`;
          }
        }
      });

      totalDumpsEl.textContent = totalDumps;
    } catch (error) {
      console.error('Error fetching cameras:', error);
      cameraContainer.innerHTML = '<div class="loading-cameras"><p>Error connecting to server.</p></div>';
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
