import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function createViewer(container) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: "low-power",
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setClearColor(0x080909);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  container.replaceChildren(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.minDistance = 1.6;
  controls.maxDistance = 12;
  controls.maxPolarAngle = Math.PI * 0.9;
  controls.listenToKeyEvents(container);
  scene.add(new THREE.HemisphereLight(0xfff4e5, 0x35444a, 2.1));
  const key = new THREE.DirectionalLight(0xffedd9, 3.2);
  key.position.set(-3, 5, 5);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xcbe0ff, 2.2);
  rim.position.set(4, 2, -2);
  scene.add(rim);
  const group = new THREE.Group();
  scene.add(group);
  let body, strands, controller, info;
  function render() {
    if (!container.hidden && container.clientWidth)
      renderer.render(scene, camera);
  }
  function resize() {
    const width = container.clientWidth,
      height = container.clientHeight;
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    render();
  }
  new ResizeObserver(resize).observe(container);
  controls.addEventListener("change", render);
  renderer.domElement.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
  });
  renderer.domElement.addEventListener("webglcontextrestored", render);
  function clear() {
    for (const object of [...group.children]) {
      object.geometry.dispose();
      object.material.dispose();
      group.remove(object);
    }
  }
  function reset() {
    const aspect = container.clientWidth / Math.max(1, container.clientHeight);
    // Preview coordinates preserve the approved pose, with world Y as vertical.
    const low = new THREE.Vector3(...info.bounds[0]);
    const high = new THREE.Vector3(...info.bounds[1]);
    const size = high.clone().sub(low);
    const center = high.clone().add(low).multiplyScalar(0.5);
    const distance =
      (Math.max(size.y, size.x / aspect) /
        (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) *
        1.22 +
      size.z * 0.5;
    controls.target.copy(center);
    camera.position
      .copy(center)
      .add(new THREE.Vector3(0, distance * 0.09, distance));
    controls.update();
    resize();
  }
  return {
    cancel() {
      controller?.abort();
    },
    async load(id) {
      controller?.abort();
      const current = new AbortController();
      controller = current;
      const response = await fetch(`assets/models/${id}.json`, {
        signal: current.signal,
      });
      if (!response.ok) throw new Error(`Missing ${id} model metadata`);
      const metadata = await response.json();
      const binary = await fetch(metadata.url, { signal: current.signal });
      if (!binary.ok) throw new Error(`Missing ${id} geometry`);
      const buffer = await binary.arrayBuffer();
      if (current.signal.aborted)
        throw new DOMException("Aborted", "AbortError");
      if (buffer.byteLength !== metadata.bytes)
        throw new Error("Incomplete geometry download");
      const view = (name, Type) =>
        new Type(
          buffer,
          metadata.layout[name].offset,
          metadata.layout[name].length,
        );
      clear();
      info = metadata;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(view("bodyPositions", Float32Array), 3),
      );
      geometry.setIndex(
        new THREE.BufferAttribute(view("bodyIndices", Uint32Array), 1),
      );
      geometry.computeVertexNormals();
      body = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: new THREE.Color(...metadata.bodyColor),
          roughness: 0.73,
          metalness: 0,
        }),
      );
      group.add(body);
      const hair = new THREE.BufferGeometry();
      const positions = view("strandPositions", Float32Array);
      hair.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const points = metadata.pointsPerStrand;
      const indices = new Uint32Array(
        metadata.previewStrands * (points - 1) * 2,
      );
      let write = 0;
      for (let i = 0; i < metadata.previewStrands; i++)
        for (let j = 0; j < points - 1; j++) {
          indices[write++] = i * points + j;
          indices[write++] = i * points + j + 1;
        }
      hair.setIndex(new THREE.BufferAttribute(indices, 1));
      strands = new THREE.LineSegments(
        hair,
        new THREE.LineBasicMaterial({
          color: new THREE.Color(...metadata.hairColor),
          transparent: false,
        }),
      );
      group.add(strands);
      container.dataset.loadedAnimal = id;
      reset();
      render();
      return metadata;
    },
    setVisible(key, visible) {
      const object = key === "body" ? body : strands;
      if (object) object.visible = visible;
      render();
    },
    reset,
    get metadata() {
      return info;
    },
  };
}
