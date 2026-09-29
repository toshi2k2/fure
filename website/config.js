export const site = {
  title:
    "FurE: Efficient Instance-Specific 3D Fur Reconstruction without Animal-Fur Datasets",
  paper: "assets/paper/FurE.pdf",
  code: "https://github.com/toshi2k2/fure",
  video: "assets/video/fox-strong-wind-4k60.mp4",
  videoFallback: "assets/video/fox-strong-wind-4k60.webm",
  authorInterval: 5000,
  firstAuthors: [
    {
      name: "Srinjay Sarkar",
      email: "ssarka29@jh.edu",
      marks: "★",
      description: "Equal contribution",
    },
    {
      name: "Prakhar Kaushik",
      email: "pkaushi1@jh.edu",
      marks: "★†",
      description: "Equal contribution, project lead",
    },
  ],
  animals: [
    {
      id: "cat",
      name: "Cat",
      source: "Artemis",
      strands: 310000,
      description:
        "Fine strands, a full coat, and a distinct silhouette. Inspect the explicit geometry beneath the studio rendering.",
    },
    {
      id: "fox",
      name: "Fox",
      source: "Artemis",
      strands: 265000,
      description:
        "From short facial fur to a full, bushy tail. The same reconstructed groom appears in the wind demonstration below.",
    },
    {
      id: "tiger",
      name: "Tiger",
      source: "Artemis",
      strands: 265000,
      description:
        "A dense coat reconstructed as individual curves, rather than a texture painted onto an outer surface.",
    },
    {
      id: "beagle",
      name: "Beagle",
      source: "Artemis",
      strands: 100000,
      description:
        "Short fur follows the body. Strand roots, local orientation, and part-level length guide the reconstructed coat.",
    },
    {
      id: "panda",
      name: "Panda",
      source: "Artemis",
      strands: 220000,
      description:
        "A thick coat with explicit strand geometry, ready for inspection, editing, and downstream rendering.",
    },
    {
      id: "bison",
      name: "Bison",
      source: "Real-world capture",
      strands: 104993,
      description:
        "From real multiview photographs to an editable fur asset. A challenging coat with fine facial regions and a heavy mane.",
    },
  ],
};

export const bibtex = `@misc{fure,
  title  = {FurE: Efficient Instance-Specific 3D Fur
            Reconstruction without Animal-Fur Datasets},
  author = {Sarkar, Srinjay and Kaushik, Prakhar and
            Paul, Soumava and Yuille, Alan},
  year   = {TODO},
  note   = {Placeholder: publication details forthcoming},
  url    = {https://toshi2k2.github.io/fure/}
}`;
