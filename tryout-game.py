import axelrod as axl
from collections import namedtuple
from numpy import median
import axelrod as axl
import csv
import tqdm

import os


def label(prefix, results):
    """
    A label used for the various plots
    """
    return "{} - turns: {}, repetitions: {}, strategies: {}. ".format(prefix,
                turns, results.repetitions, results.num_players)

def obtain_assets(results, strategies_name="strategies",
                  tournament_type="std",
                  assets_dir="./assets", lengthplot=False):

    total = 6 + int(lengthplot)

    pbar = tqdm.tqdm(total=total, desc="Obtaining plots")

    file_path_root = "{}/{}_{}".format(assets_dir, strategies_name,
                                       tournament_type)
    plot = axl.Plot(results)

    f = plot.boxplot(title=label("Payoff", results))
    f.savefig("{}_boxplot.svg".format(file_path_root))
    pbar.update()

    f = plot.payoff(title=label("Payoff", results))
    f.savefig("{}_payoff.svg".format(file_path_root))
    pbar.update()

    f = plot.winplot(title=label("Wins", results))
    f.savefig("{}_winplot.svg".format(file_path_root))
    pbar.update()

    f = plot.sdvplot(title=label("Payoff differences", results))
    f.savefig("{}_sdvplot.svg".format(file_path_root))
    pbar.update()

    f = plot.pdplot(title=label("Payoff differences", results))
    f.savefig("{}_pdplot.svg".format(file_path_root))
    pbar.update()

    eco = axl.Ecosystem(results)
    eco.reproduce(1000)
    f = plot.stackplot(eco, title=label("Eco", results))
    f.savefig("{}_reproduce.svg".format(file_path_root))
    pbar.update()

    if lengthplot is True:
        f = plot.lengthplot(title=label("Length of matches", results))
        f.savefig("{}_lengthplot.svg".format(file_path_root))
        pbar.update()


players = [s() for s in axl.test_strategies]  # Create players

turns = 5
repetitions = 5

processes = 2
seed = 1
filename = "data/test_tournament.csv"

def main(players=players):
    # Deleting the file if it exists
    try:
        os.remove(filename)
    except OSError:
        pass

    tournament = axl.Tournament(players, turns=turns, repetitions=repetitions, seed=seed)

    results = tournament.play(filename=filename)
    results.write_summary('assets/test_summary.csv')

if __name__ == "__main__":
    main()

#processes = 20
#players = [s() for s in axl.test_strategies]  # Create players
#filename = "data/test_tournament.csv"
#tournament = axl.Tournament(players, seed=1)  # Create a tournament
#results = tournament.play(filename=filename, processes=processes)
#results.write_summary('data/test_summary.csv')

