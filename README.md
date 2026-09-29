# fure
Official website for FurE: Efficient Instance-Specific 3D Fur Reconstruction without Animal-Fur Datasets.

Website source, self-contained media, build instructions and asset provenance are
in [website/](website/README.md). The supplied LaTeX draft is in `paper/`.

```sh
cd website
npm ci
npm run build
npm run serve
```

Open `http://127.0.0.1:4173`.

## Publish The Website, Not This README

1. Open [Settings > Pages](https://github.com/toshi2k2/fure/settings/pages).
2. Under **Build and deployment > Source**, select **GitHub Actions**, not
   **Deploy from a branch**. Do not select `main / (root)`.
3. Open [Actions > Deploy FurE website](https://github.com/toshi2k2/fure/actions/workflows/website.yml),
   choose **Run workflow**, select `main`, and run it.
4. After the build and deploy jobs succeed, open
   [the project website](https://toshi2k2.github.io/fure/).

The existing workflow builds the site and publishes **only `website/dist/`**.
It does not publish the root README, LaTeX source directory, or agent notes.
If the homepage shows this README, Pages is using the repository-root Jekyll
build instead of the intended workflow. `.gitignore` does not choose the Pages
publishing source. A push alone does not trigger our manual-only workflow.
