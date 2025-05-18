# AlphaZeroFromScratch
Out of the box implementation based on the code of the tutorial: [AlphaZero](https://github.com/foersterrobert/AlphaZero)

![tictactoe](https://raw.githubusercontent.com/foersterrobert/AlphaZero/master/assets/tictactoe.gif)
![connectfour](https://raw.githubusercontent.com/foersterrobert/AlphaZero/master/assets/connectfour.gif)

### Some Helpful Resources
* AlphaZero-Paper: https://arxiv.org/pdf/1712.01815.pdf
* Paper-Walkthrough: https://youtu.be/0slFo1rV0EM
* MCTS-Explained: https://youtu.be/UXW2yZndl7U
* AlphaZero-Explained: https://youtu.be/62nq4Zsn8vc

## Python project

The notebook `8.ConnectFour.ipynb` has been converted into standalone Python modules.
The main classes live inside the `alphazero` package and example scripts are
available in the `scripts` folder.

Run training:

```bash
python3 scripts/train_connect_four.py
```

Play against the AI (random weights by default):

```bash
python3 scripts/play_connect_four.py
```

❤

