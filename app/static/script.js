const pdfFile = document.getElementById("pdfFile");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const askStatus = document.getElementById("askStatus");
const answerCard = document.getElementById("answerCard");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");

function setStatus(el, message, kind) {
  el.textContent = message;
  el.className = "status" + (kind ? " " + kind : "");
}

function enableQA(enabled) {
  questionInput.disabled = !enabled;
  askBtn.disabled = !enabled;
}

uploadBtn.addEventListener("click", async () => {
  const file = pdfFile.files && pdfFile.files[0];
  if (!file) {
    setStatus(uploadStatus, "Please select a PDF file first.", "error");
    return;
  }
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    setStatus(uploadStatus, "Invalid file type. Only PDF files are allowed.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  setStatus(uploadStatus, "Uploading and processing document...", "");
  uploadStatus.classList.add("spinner");

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "Upload failed.");
    }
    setStatus(
      uploadStatus,
      `Done: ${data.filename} processed into ${data.chunks} chunks. You can now ask questions.`,
      "success"
    );
    enableQA(true);
  } catch (err) {
    setStatus(uploadStatus, err.message || "Upload failed.", "error");
    enableQA(false);
  } finally {
    uploadBtn.disabled = false;
    uploadStatus.classList.remove("spinner");
  }
});

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus(askStatus, "Please enter a question.", "error");
    return;
  }
  askBtn.disabled = true;
  setStatus(askStatus, "Thinking...", "");
  askStatus.classList.add("spinner");
  answerCard.classList.add("hidden");

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "Failed to get an answer.");
    }
    answerEl.textContent = data.answer || "(empty answer)";
    sourcesEl.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
      data.sources.forEach((s) => {
        const div = document.createElement("div");
        div.className = "source";
        const title = document.createElement("div");
        title.className = "source-title";
        title.textContent = `Page ${s.page}`;
        const body = document.createElement("div");
        body.className = "source-body";
        body.textContent = s.content;
        div.appendChild(title);
        div.appendChild(body);
        sourcesEl.appendChild(div);
      });
    } else {
      sourcesEl.textContent = "No sources returned.";
    }
    answerCard.classList.remove("hidden");
    setStatus(askStatus, "", "");
  } catch (err) {
    setStatus(askStatus, err.message || "Failed to get an answer.", "error");
  } finally {
    askBtn.disabled = false;
    askStatus.classList.remove("spinner");
  }
}

askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});
