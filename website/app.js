import { site, bibtex } from "./config.js";

const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let authorsPaused = reducedMotion.matches;
const authorGroup = $("first-authors");
const authorPause = $("author-pause");
function updatePauseLabel() {
  authorPause.setAttribute("aria-pressed", String(authorsPaused));
  authorPause.textContent = authorsPaused
    ? "Resume author order ▷"
    : "Pause author order Ⅱ";
}
updatePauseLabel();
authorPause.addEventListener("click", () => {
  authorsPaused = !authorsPaused;
  updatePauseLabel();
});
reducedMotion.addEventListener("change", ({ matches }) => {
  if (matches) {
    authorsPaused = true;
    updatePauseLabel();
  }
});
function rotateAuthors() {
  if (
    authorsPaused ||
    document.hidden ||
    authorGroup.matches(":hover, :focus-within")
  )
    return;
  authorGroup.append(authorGroup.firstElementChild);
  if (!reducedMotion.matches) {
    for (const author of authorGroup.children)
      author.animate(
        [
          { opacity: 0.2, transform: "translateY(3px)" },
          { opacity: 1, transform: "none" },
        ],
        { duration: 350, easing: "ease-out" },
      );
  }
}
let authorTimer = setInterval(rotateAuthors, site.authorInterval);
window.addEventListener("pagehide", () => {
  clearInterval(authorTimer);
  authorTimer = null;
});
window.addEventListener("pageshow", () => {
  if (authorTimer === null)
    authorTimer = setInterval(rotateAuthors, site.authorInterval);
});

// The scientific presentation remains visible if scripting or WebGL is unavailable.
if ("IntersectionObserver" in window && !reducedMotion.matches) {
  const reveal = new IntersectionObserver(
    (entries) => {
      for (const entry of entries)
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          reveal.unobserve(entry.target);
        }
    },
    { threshold: 0.06 },
  );
  document.querySelectorAll(".reveal").forEach((element) => {
    if (element.getBoundingClientRect().top > innerHeight)
      element.classList.add("js-reveal");
    reveal.observe(element);
  });
}

let selected = site.animals[0];
let mode = "studio";
let viewer = null;
let viewerPromise = null;
let requestId = 0;
const tabs = [...document.querySelectorAll("[data-animal]")];
const studioButton = $("studio-mode");
const interactiveButton = $("interactive-mode");

async function setMode(nextMode) {
  mode = nextMode;
  const token = ++requestId;
  studioButton.setAttribute("aria-pressed", String(mode === "studio"));
  interactiveButton.setAttribute(
    "aria-pressed",
    String(mode === "interactive"),
  );
  $("viewer-loading").hidden = true;
  if (mode === "studio") {
    viewer?.cancel();
    $("webgl-viewer").hidden = true;
    $("animal-image").hidden = false;
    $("geometry-controls").hidden = true;
    $("viewer-note").textContent =
      "Full-density render · original teaser palette";
    return;
  }
  $("viewer-loading").textContent = "Loading the 3D preview…";
  $("viewer-loading").hidden = false;
  try {
    if (!viewerPromise)
      viewerPromise = import("./viewer.js").then(({ createViewer }) =>
        createViewer($("webgl-viewer")),
      );
    viewer = await viewerPromise;
    if (token !== requestId) return;
    $("webgl-viewer").hidden = false;
    $("animal-image").hidden = true;
    const metadata = await viewer.load(selected.id);
    if (token !== requestId) return;
    $("viewer-loading").hidden = true;
    $("geometry-controls").hidden = false;
    $("viewer-note").textContent =
      `${metadata.previewStrands.toLocaleString()} sampled strands · drag to orbit / pinch to zoom`;
    $("show-body").setAttribute("aria-pressed", "true");
    $("show-strands").setAttribute("aria-pressed", "true");
  } catch (error) {
    if (token !== requestId || error.name === "AbortError") return;
    console.info("3D preview unavailable:", error.message);
    // A failed asset fetch should not create a second WebGL context on retry.
    if (!viewer) viewerPromise = null;
    await setMode("studio");
    $("viewer-note").textContent =
      "3D unavailable on this device. Full-density studio view shown.";
  }
}

function selectAnimal(id, focus = false) {
  selected = site.animals.find((animal) => animal.id === id);
  if (!selected) return;
  for (const tab of tabs) {
    const active = tab.dataset.animal === id;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  }
  $("animal-panel").setAttribute("aria-labelledby", `tab-${id}`);
  $("animal-image").src = `assets/images/${id}.webp`;
  $("animal-image").alt =
    `Full-density studio rendering of the reconstructed ${selected.name.toLowerCase()} and its underlying body.`;
  $("animal-name").textContent = selected.name;
  $("animal-source").textContent = selected.source;
  $("animal-description").textContent = selected.description;
  $("strand-count").textContent = selected.strands.toLocaleString("en-US");
  $("animal-index").textContent =
    `${String(site.animals.indexOf(selected) + 1).padStart(2, "0")} / 06`;
  if (mode === "interactive") setMode("interactive");
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectAnimal(tab.dataset.animal));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft")
      next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      selectAnimal(tabs[next].dataset.animal, true);
    }
  });
});
studioButton.addEventListener("click", () => setMode("studio"));
interactiveButton.addEventListener("click", () => setMode("interactive"));
for (const key of ["body", "strands"])
  $("show-" + key).addEventListener("click", (event) => {
    const button = event.currentTarget;
    const visible = button.getAttribute("aria-pressed") !== "true";
    viewer?.setVisible(key, visible);
    button.setAttribute("aria-pressed", String(visible));
  });
$("reset-view").addEventListener("click", () => viewer?.reset());
$("viewer-fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if ($("viewer-stage").requestFullscreen)
      await $("viewer-stage").requestFullscreen();
    else window.open($("animal-image").src, "_blank", "noopener");
  } catch {
    $("viewer-note").textContent = "Fullscreen is unavailable in this browser.";
  }
});
$("inspect-bison").addEventListener("click", () => {
  selectAnimal("bison");
  $("collection").scrollIntoView({
    behavior: reducedMotion.matches ? "instant" : "smooth",
  });
  $("tab-bison").focus({ preventScroll: true });
});

const video = $("wind-video");
const playButton = $("play-film");
playButton.addEventListener("click", async () => {
  playButton.disabled = true;
  try {
    if (!video.getAttribute("src")) {
      video.src = video.canPlayType('video/mp4; codecs="avc1.640034"')
        ? site.video
        : site.videoFallback;
      video.load();
    }
    try {
      await video.play();
    } catch (error) {
      if (
        video.src.endsWith(".mp4") &&
        video.canPlayType('video/webm; codecs="vp9"')
      ) {
        video.src = site.videoFallback;
        video.load();
        await video.play();
      } else throw error;
    }
    playButton.hidden = true;
    video.closest(".film").classList.add("is-playing");
  } catch {
    playButton.querySelector("span:last-child").textContent =
      "Playback unavailable. Download the 4K film below.";
  } finally {
    playButton.disabled = false;
  }
});
if ("IntersectionObserver" in window) {
  new IntersectionObserver(
    ([entry]) => {
      if (!entry.isIntersecting && !video.paused) video.pause();
    },
    { threshold: 0.05 },
  ).observe(video);
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) video.pause();
});

$("bibtex").textContent = bibtex;
$("copy-bibtex").addEventListener("click", async () => {
  const button = $("copy-bibtex");
  try {
    await navigator.clipboard.writeText(bibtex);
    button.textContent = "Copied ✓";
    $("copy-status").textContent =
      "Placeholder BibTeX copied to the clipboard.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents($("bibtex"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = "Selected. Press Ctrl/Cmd+C";
    $("copy-status").textContent =
      "Automatic copy unavailable. Citation selected for manual copying.";
  }
  setTimeout(() => {
    button.textContent = "Copy citation ⧉";
  }, 3500);
});
