const imageInput = document.getElementById("imageInput");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const dropZone = document.getElementById("drop-zone");
const dropPrompt = document.getElementById("drop-prompt");
const fileHint = document.getElementById("file-hint");
const clearButton = document.getElementById("clear-button");
const statusMessage = document.getElementById("status");
const resultEmpty = document.getElementById("result-empty");
const prediction = document.getElementById("prediction");
const predictionLabel = document.getElementById("label");
const confidenceText = document.getElementById("confidence-text");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceProgress = document.getElementById("confidence-progress");
const meaningCard = document.getElementById("meaning-card");
const flowerMeaning = document.getElementById("flower-meaning");
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.4;
const flowerNames = ["ดอกมะลิ", "ดอกกุหลาบ", "ดอกทานตะวัน"];
const flowerMeanings = [
  "สื่อถึงความบริสุทธิ์ ความรักที่อ่อนโยน และความกตัญญู มักใช้แทนความรักและความปรารถนาดีที่มีให้คนสำคัญ",
  "สื่อถึงความรัก ความโรแมนติก และความหลงใหล โดยความหมายอาจแตกต่างกันไปตามสีของดอก",
  "สื่อถึงความสุข ความสดใส ความหวัง และความภักดี เปรียบเหมือนการส่งพลังบวกและกำลังใจให้กัน",
];

let session = null;
let selectedImageUrl = null;
let modelReady = false;

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`.trim();
}

async function loadModel() {
  try {
    if (!window.ort) throw new Error("โหลด ONNX Runtime ไม่สำเร็จ");
    session = await ort.InferenceSession.create("flower.onnx", {
      executionProviders: ["wasm"],
    });
    modelReady = true;
    setStatus("พร้อมทำนายแล้ว เลือกรูปดอกไม้ได้เลย", "success");
  } catch (error) {
    console.error("Unable to load flower model:", error);
    setStatus("โหลดโมเดลไม่สำเร็จ ลองเปิดเว็บผ่าน Live Server หรือ local server แล้วรีเฟรช", "error");
  }
}

function preprocessImage(image) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = INPUT_SIZE;
  tempCanvas.height = INPUT_SIZE;
  const tempContext = tempCanvas.getContext("2d", { willReadFrequently: true });
  tempContext.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);

  const pixels = tempContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  const values = new Float32Array(3 * planeSize);

  for (let i = 0; i < planeSize; i += 1) {
    values[i] = pixels[i * 4] / 255;
    values[i + planeSize] = pixels[i * 4 + 1] / 255;
    values[i + planeSize * 2] = pixels[i * 4 + 2] / 255;
  }

  return new ort.Tensor("float32", values, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function decodePredictions(output, threshold = CONFIDENCE_THRESHOLD) {
  const [batchSize, secondDimension, thirdDimension] = output.dims;
  const classCount = flowerNames.length;
  const channelsFirst = secondDimension === classCount + 4;
  const attributes = channelsFirst ? secondDimension : thirdDimension;
  const boxCount = channelsFirst ? thirdDimension : secondDimension;
  const data = output.data;
  const boxes = [];

  if (batchSize !== 1 || attributes < classCount + 4) {
    throw new Error(`รูปแบบผลลัพธ์โมเดลไม่ตรงที่คาดไว้: ${output.dims.join(" × ")}`);
  }

  const valueAt = (boxIndex, attributeIndex) => channelsFirst
    ? data[attributeIndex * boxCount + boxIndex]
    : data[boxIndex * attributes + attributeIndex];

  for (let i = 0; i < boxCount; i += 1) {
    let bestScore = 0;
    let classId = 0;

    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const score = valueAt(i, classIndex + 4);
      if (score > bestScore) {
        bestScore = score;
        classId = classIndex;
      }
    }

    if (bestScore < threshold) continue;

    const centerX = valueAt(i, 0);
    const centerY = valueAt(i, 1);
    const width = valueAt(i, 2);
    const height = valueAt(i, 3);
    boxes.push({
      x: centerX - width / 2,
      y: centerY - height / 2,
      width,
      height,
      score: bestScore,
      classId,
    });
  }

  return boxes;
}

function intersectionOverUnion(a, b) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function suppressOverlaps(boxes) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const selected = [];

  while (sorted.length) {
    const current = sorted.shift();
    selected.push(current);
    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (sorted[i].classId === current.classId && intersectionOverUnion(current, sorted[i]) >= IOU_THRESHOLD) {
        sorted.splice(i, 1);
      }
    }
  }

  return selected;
}

function showPrediction(boxes) {
  resultEmpty.hidden = true;
  prediction.hidden = false;

  if (!boxes.length) {
    predictionLabel.textContent = "ยังระบุชนิดไม่ได้";
    meaningCard.hidden = true;
    confidenceText.textContent = "ลองใหม่";
    confidenceBar.style.width = "0%";
    confidenceProgress.setAttribute("aria-valuenow", "0");
    setStatus("ไม่พบดอกไม้ที่โมเดลรู้จัก ลองใช้ภาพที่เห็นดอกไม้ชัดขึ้น", "error");
    return;
  }

  const best = boxes.reduce((highest, box) => (box.score > highest.score ? box : highest));
  const percent = Math.max(0, Math.min(100, Math.round(best.score * 100)));
  predictionLabel.textContent = flowerNames[best.classId] ?? "ไม่ทราบชนิด";
  flowerMeaning.textContent = flowerMeanings[best.classId] ?? "ยังไม่มีข้อมูลความหมายของดอกไม้ชนิดนี้";
  meaningCard.hidden = false;
  confidenceText.textContent = `${percent}%`;
  confidenceBar.style.width = `${percent}%`;
  confidenceProgress.setAttribute("aria-valuenow", String(percent));
  setStatus(`พบดอกไม้ ${boxes.length} ตำแหน่ง`, "success");
}

function drawDetections(boxes) {
  ctx.lineWidth = 3;
  ctx.font = "600 18px Prompt, sans-serif";
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const x = Math.max(0, box.x);
    const y = Math.max(0, box.y);
    const width = Math.min(canvas.width - x, box.width);
    const height = Math.min(canvas.height - y, box.height);
    const caption = `${flowerNames[box.classId] ?? "ดอกไม้"} ${Math.round(box.score * 100)}%`;
    const captionWidth = ctx.measureText(caption).width + 16;
    const captionY = Math.max(0, y - 30);

    ctx.strokeStyle = "#75b879";
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = "#355a45";
    ctx.fillRect(x, captionY, captionWidth, 27);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(caption, x + 8, captionY + 4);
  });
}

async function predictImage(image) {
  if (!modelReady) {
    setStatus("โมเดลยังโหลดไม่เสร็จ รอสักครู่แล้วลองอีกครั้ง", "error");
    return;
  }

  setStatus("กำลังวิเคราะห์ภาพ...", "");
  clearButton.disabled = true;

  try {
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    ctx.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const tensor = preprocessImage(image);
    const inputName = session.inputNames[0];
    const output = await session.run({ [inputName]: tensor });
    const outputTensor = output[session.outputNames[0]];
    const boxes = suppressOverlaps(decodePredictions(outputTensor));

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    drawDetections(boxes);
    showPrediction(boxes);
  } catch (error) {
    console.error("Flower prediction failed:", error);
    setStatus("วิเคราะห์ภาพไม่สำเร็จ ลองเลือกรูปอื่นอีกครั้ง", "error");
  } finally {
    clearButton.disabled = false;
  }
}

function handleImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("ไฟล์นี้ไม่ใช่รูปภาพ กรุณาเลือกไฟล์ภาพ", "error");
    return;
  }

  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    dropPrompt.hidden = true;
    canvas.style.display = "block";
    clearButton.disabled = false;
    fileHint.textContent = file.name;
    resultEmpty.hidden = false;
    prediction.hidden = true;
    predictImage(image);
    URL.revokeObjectURL(selectedImageUrl);
    selectedImageUrl = null;
  };
  image.onerror = () => setStatus("เปิดรูปภาพนี้ไม่ได้ กรุณาเลือกไฟล์อื่น", "error");
  image.src = selectedImageUrl;
}

imageInput.addEventListener("change", (event) => handleImageFile(event.target.files[0]));

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", (event) => {
  if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  handleImageFile(event.dataTransfer.files[0]);
});

clearButton.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.style.display = "none";
  dropPrompt.hidden = false;
  imageInput.value = "";
  clearButton.disabled = true;
  fileHint.textContent = "แนะนำให้ใช้ภาพดอกไม้ชัด ๆ และมีแสงสว่างเพียงพอ";
  prediction.hidden = true;
  meaningCard.hidden = true;
  resultEmpty.hidden = false;
  setStatus(modelReady ? "พร้อมทำนายแล้ว เลือกรูปดอกไม้ได้เลย" : "กำลังเตรียมระบบทำนาย...", modelReady ? "success" : "");
});

loadModel();
