import torch
from alphazero.connect_four import ConnectFour
from alphazero.net import ResNet
from alphazero.alpha_zero import AlphaZero


def main():
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, num_res_blocks=9, num_hidden=128, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_selfPlay_iterations': 500,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
    }

    alpha_zero = AlphaZero(model, optimizer, game, args)
    alpha_zero.learn()


if __name__ == "__main__":
    main()

