# MLP 2, Electric Boogaloo
Basically a repeat of my first mlp mnist solver (https://github.com/benlin-gan/my-first-nn/) but written in Haskell this time. Mostly just an excuse for me to learn Haskell.
## Compiling
- run the `compile.sh` script 
- or run `ghc -dynamic -O mlp-mnist`
- uses dynamic linking because otherwise it doesn't work on arch linux 
## Notes on "784\_16\_16\_10\_sig.mdl"
- 1/(1 + e^-x) activation
- learning rate 0.1
- initialization between -1 and 1 using (mkStdGen 523)
- run `./mlp-mnist stats` to see its accuracy
## Miscellaneous Thoughts on Haskell
- Coding this project was a way better experience in Haskell than Rust
	- because Rust isn't really meant for this kind of project though lol.
- Also I ran into annoying memory issues with the combination of the garbage collector, Haskell laziness, and my use of the `foldl (>=>) return` pattern.
