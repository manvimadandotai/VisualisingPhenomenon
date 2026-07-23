# VisualisingPhenomenon
Creating a repository to visualize intellectually rich physics or mathematical phenomenon.

## The Energy Cascade

`index.html` is a self-contained, interactive visualization of the turbulent energy cascade — Kolmogorov's picture of kinetic energy pouring from large eddies to small ones, and L. F. Richardson's 1922 verse ("big whirls have little whirls that feed on their velocity...").

Each "stir" of the fluid plays out its own arc — **birth** (a large eddy is injected), **growth** (it splits into a tree of smaller, faster-spinning eddies, generation after generation), **breakdown** (at the smallest scales, eddies dissipate into heat) — before the fluid stills and the next stir begins. A live log-log energy spectrum panel bins the active eddies by scale, comparing the simulation against Kolmogorov's −5/3 law in real time. Click anywhere on the canvas to stir in energy yourself.

No build step or dependencies — open `index.html` directly in a browser, or serve the repo with GitHub Pages.
