import axelrod as axl
players = [s() for s in axl.test_strategies]  # Create players
tournament = axl.Tournament(players, seed=1)  # Create a tournament
results = tournament.play()  # Play the tournament
results.ranked_names
['Defector', 'Grudger', 'Tit For Tat', 'Cooperator', 'Random: 0.5', 'MediocreQLearner','HesitantQLearner','CautiousQLearner', 'DeepQLearner']



