document.addEventListener("DOMContentLoaded", async () => {
  const CONFIDENCE_WARNING_THRESHOLD = 72;

  const dropZone = document.getElementById("dropZone");
  const imageInput = document.getElementById("imageInput");
  const browseBtn = document.getElementById("browseBtn");
  const predictBtn = document.getElementById("predictBtn");
  const selectedFileLabel = document.getElementById("selectedFile");
  const loaderWrap = document.getElementById("loaderWrap");
  const errorBanner = document.getElementById("errorBanner");

  const previewPanel = document.getElementById("previewPanel");
  const previewImg = document.getElementById("previewImg");

  const resultsPanel = document.getElementById("resultsPanel");
  const summaryPanel = document.getElementById("summaryPanel");
  const influencePanel = document.getElementById("influencePanel");

  const originalImg = document.getElementById("originalImg");
  const enhancedImg = document.getElementById("enhancedImg");
  const heatmapImg = document.getElementById("heatmapImg");
  const focusImg = document.getElementById("focusImg");
  const resultText = document.getElementById("resultText");
  const labelText = document.getElementById("labelText");
  const confidenceText = document.getElementById("confidenceText");
  const explanationText = document.getElementById("explanationText");
  const statusChip = document.getElementById("statusChip");

  let selectedFile = null;
  let previewUrl = null;

  function setError(message) {
    errorBanner.textContent = message || "";
    errorBanner.hidden = !message;
  }

  function setLoading(isLoading) {
    loaderWrap.hidden = !isLoading;
    predictBtn.disabled = isLoading || !selectedFile;
    browseBtn.disabled = isLoading;
    predictBtn.textContent = isLoading ? "Analyzing..." : "Start Analysis";
  }

  function updateBadge(type, text) {
    statusChip.className = `result-badge ${type}`;
    statusChip.textContent = text;
  }

  function hideResults() {
    resultsPanel.hidden = true;
    summaryPanel.hidden = true;
    influencePanel.hidden = true;
    resultsPanel.classList.remove("visible");
    summaryPanel.classList.remove("visible");
    influencePanel.classList.remove("visible");
  }

  function resetResultContent() {
    resultText.textContent = "-";
    labelText.textContent = "-";
    confidenceText.textContent = "-";
    explanationText.textContent = "The explanation for this scan will appear here after analysis.";
    updateBadge("neutral", "Analysis complete");
  }

  function showResults() {
    resultsPanel.hidden = false;
    summaryPanel.hidden = false;
    influencePanel.hidden = false;
    requestAnimationFrame(() => {
      resultsPanel.classList.add("visible");
      summaryPanel.classList.add("visible");
      influencePanel.classList.add("visible");
    });
  }

  function applyRiskState(label, confidence) {
    if (confidence < CONFIDENCE_WARNING_THRESHOLD) {
      updateBadge("warning", "Review recommended");
      return;
    }

    if ((label || "").toLowerCase() === "normal") {
      updateBadge("normal", "Normal");
    } else {
      updateBadge("risk", "Risk detected");
    }
  }

  function handleSelectedFile(file) {
    const validTypes = ["image/png", "image/jpeg", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      setError("Please upload a valid X-ray image");
      return;
    }

    selectedFile = file;
    selectedFileLabel.textContent = file.name;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    previewPanel.hidden = false;
    predictBtn.disabled = false;

    hideResults();
    resetResultContent();
    setError("");
  }

  async function updateSystemStatus() {
    try {
      const response = await fetch("/health");
      const payload = await response.json();
      if (!payload.ready) {
        setError("Unable to process image. Please try again");
      }
    } catch (error) {
      console.error(error);
    }
  }

  dropZone.addEventListener("click", () => imageInput.click());
  browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    imageInput.click();
  });

  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    if (file) {
      handleSelectedFile(file);
    }
  });

  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
      handleSelectedFile(file);
    }
  });

  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      setError("Please upload a valid X-ray image");
      return;
    }

    setLoading(true);
    setError("");
    hideResults();

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Unable to process image. Please try again");
      }

      const confidence = Number(payload.confidence || 0);
      originalImg.src = payload.original_image;
      enhancedImg.src = payload.enhanced_image;
      heatmapImg.src = payload.heatmap_image;
      focusImg.src = payload.heatmap_image;
      resultText.textContent = payload.result;
      labelText.textContent = payload.label;
      confidenceText.textContent = `${confidence}%`;
      explanationText.textContent = payload.explanation;
      applyRiskState(payload.label, confidence);
      showResults();
    } catch (error) {
      setError(error.message || "Unable to process image. Please try again");
    } finally {
      setLoading(false);
    }
  });

  previewPanel.hidden = true;
  hideResults();
  resetResultContent();
  await updateSystemStatus();
});
