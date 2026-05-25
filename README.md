# dialectical-othello

Dialectical Othello engine on the `dialectical-games` core. Third game
cartridge (after chess + checkers).

## Architecture

Game-agnostic argumentation, search, and Protocols live in
[`ctoth/dialectical-games`](https://github.com/ctoth/dialectical-games).
This repo will implement the Othello-specific Board + Move types and the
cartridge wiring on top of that core.

## Sibling repos

- Core: [`ctoth/dialectical-games`](https://github.com/ctoth/dialectical-games)
- Chess cartridge: [`ctoth/dialectical-chess`](https://github.com/ctoth/dialectical-chess)
- Checkers cartridge: [`ctoth/dialectical-checkers`](https://github.com/ctoth/dialectical-checkers)

## Status

Scaffolding; Othello cartridge implementation in progress.
