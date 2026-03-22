const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const debugBtn = document.getElementById("debugBtn");
const parseBtn = document.getElementById("parseBtn");

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

async function runParse(isDebug) {
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Please select a file.");
    return;
  }

  // Disable buttons during request.
  if (parseBtn) parseBtn.disabled = true;
  if (debugBtn) debugBtn.disabled = true;

  setStatus(isDebug ? "Debug parsing..." : "Parsing... this may take a moment.");
  resultEl.textContent = "";

  const fd = new FormData();
  fd.append("file", file);

  const endpoint = isDebug ? "/api/parse-debug" : "/api/parse";
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }

    const data = await res.json();
    resultEl.textContent = JSON.stringify(data, null, 2);
    setStatus("Done.");
  } catch (err) {
    setStatus("Error: " + (err && err.message ? err.message : String(err)));
  } finally {
    if (parseBtn) parseBtn.disabled = false;
    if (debugBtn) debugBtn.disabled = false;
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await runParse(false);
});

if (debugBtn) {
  debugBtn.addEventListener("click", async () => {
    await runParse(true);
  });
}

