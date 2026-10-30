const state = {
  device: "unknown",
  kind: "image",
  generating: false,
  imageModels: [],
  videoModels: [],
};

async function api(path, options) {
  const res = await fetch(path, options || {});
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    data = text;
  }
  if (!res.ok) {
    throw new Error(typeof data === "string" ? data : JSON.stringify(data));
  }
  return data;
}

function el(id) {
  return document.getElementById(id);
}

function setStatus(kind, msg) {
  el("status-" + kind).textContent = msg || "";
}

function populateModels() {
  const imgSelect = el("model-image");
  const vidSelect = el("model-video");
  imgSelect.innerHTML = "";
  vidSelect.innerHTML = "";
  state.imageModels.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.id} ${m.estimated_gb ? "~" + m.estimated_gb + "GB" : ""}`;
    imgSelect.appendChild(opt);
  });
  state.videoModels.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.id} ${m.estimated_gb ? "~" + m.estimated_gb + "GB" : ""}`;
    vidSelect.appendChild(opt);
  });
  if (state.imageModels.length) el("model-image").value = state.imageModels[0].id;
  if (state.videoModels.length) el("model-video").value = state.videoModels[0].id;
  if (!state.imageModels.length && !state.videoModels.length) {
    imgSelect.innerHTML = '<option value="">no models in ./models</option>';
    vidSelect.innerHTML = '<option value="">no models in ./models</option>';
  }
}

async function refreshHealth() {
  try {
    const data = await api("/api/health");
    state.device = data.device;
    const badge = el("device-badge");
    badge.textContent = "device: " + state.device;
    badge.classList.add("online");
  } catch (e) {
    el("device-badge").textContent = "server offline";
    el("device-badge").classList.remove("online");
  }
}

async function refreshModels() {
  const data = await api("/api/models");
  state.imageModels = (data || []).filter((m) => m.kind === "image");
  state.videoModels = (data || []).filter((m) => m.kind === "video");
  populateModels();
}

async function refreshGallery() {
  const data = await api("/api/outputs");
  const gallery = el("gallery");
  gallery.innerHTML = "";
  for (const name of data || []) {
    const item = document.createElement("div");
    item.className = "gallery-item";
    const isVideo = name.toLowerCase().endsWith(".mp4");
    const media = document.createElement(isVideo ? "video" : "img");
    media.src = "/api/outputs/" + name;
    if (isVideo) {
      media.controls = true;
      media.loop = true;
      media.muted = true;
    }
    item.appendChild(media);
    gallery.appendChild(item);
  }
}

function toBase64(file) {
  if (!file) return Promise.resolve("");
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function collectParams() {
  const get = (id) => document.getElementById(id).value;
  const params = {
    prompt: el("prompt-" + state.kind).value,
    negative_prompt: el("negative-" + state.kind).value,
    width: parseInt(get("width-" + state.kind), 10) || 0,
    height: parseInt(get("height-" + state.kind), 10) || 0,
    num_inference_steps: parseInt(get("steps-" + state.kind), 10) || 25,
    guidance_scale: parseFloat(get("guidance-" + state.kind)) || 7.5,
    seed: parseInt(get("seed-" + state.kind), 10) || 0,
  };
  if (state.kind === "video") {
    params.num_frames = parseInt(get("frames-" + state.kind), 10) || 16;
    params.fps = parseInt(get("fps-" + state.kind), 10) || 7;
  }
  return params;
}

async function startJob(kind, modelId, params, initImage) {
  const body = { model: modelId, kind, ...params };
  if (initImage) {
    body.init_image = initImage;
  }
  return api("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function waitForJob(jobId) {
  for (;;) {
    const job = await api("/api/jobs/" + jobId);
    if (job.status === "done") {
      return job;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "generation failed");
    }
    await new Promise((r) => setTimeout(r, 700));
  }
}

async function generate() {
  if (state.generating) return;
  state.generating = true;
  setStatus(state.kind, "queued");
  el("generate-" + state.kind).disabled = true;
  try {
    const modelId = el("model-" + state.kind).value;
    if (!modelId) {
      throw new Error("select a model first");
    }
    const params = collectParams();
    const initImage =
      state.kind === "video" && el("init-video").files && el("init-video").files[0]
        ? await toBase64(el("init-video").files[0])
        : "";
    const job = await startJob(state.kind, modelId, params, initImage);
    setStatus(state.kind, "running");
    const done = await waitForJob(job.id);
    const resultEl = el("result-" + state.kind + "-el");
    if (state.kind === "image") {
      resultEl.src = "/api/outputs/" + done.result.image_path;
    } else {
      resultEl.src = "/api/outputs/" + done.result.video_path;
    }
    resultEl.hidden = false;
    await refreshGallery();
    setStatus(state.kind, "done");
  } catch (e) {
    setStatus(state.kind, "failed: " + (e && e.message ? e.message : String(e)));
  } finally {
    state.generating = false;
    el("generate-" + state.kind).disabled = false;
  }
}

async function refresh() {
  try {
    await Promise.all([refreshHealth(), refreshModels(), refreshGallery()]);
  } catch (e) {
    console.error(e);
  }
}

function bindSlider(sliderId, valueId) {
  const slider = el(sliderId);
  const value = el(valueId);
  slider.addEventListener("input", () => {
    value.textContent = slider.value;
  });
}

bindSlider("steps-image", "steps-value-image");
bindSlider("guidance-image", "guidance-value-image");
bindSlider("steps-video", "steps-value-video");
bindSlider("guidance-video", "guidance-value-video");
bindSlider("frames-video", "frames-value-video");
bindSlider("fps-video", "fps-value-video");

el("tab-image").addEventListener("click", () => {
  state.kind = "image";
  el("panel-image").hidden = false;
  el("panel-video").hidden = true;
  el("tab-image").classList.add("active");
  el("tab-video").classList.remove("active");
});

el("tab-video").addEventListener("click", () => {
  state.kind = "video";
  el("panel-image").hidden = true;
  el("panel-video").hidden = false;
  el("tab-video").classList.add("active");
  el("tab-image").classList.remove("active");
});

el("generate-image").addEventListener("click", generate);
el("generate-video").addEventListener("click", generate);

refresh();
setInterval(refreshHealth, 5000);
