import numpy as np
import subprocess
import dill
from pathlib import Path
from datetime import datetime

def moving_average(a, n=500):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Callback:
    def __init__(self, main_file, args):
        self.args = args

        try:
            commit_date = subprocess.check_output('git log -1 --format="%at"', shell=True, universal_newlines=True)[:-1]
            commit_date = datetime.utcfromtimestamp(int(commit_date)).strftime('UTC_%Y-%m-%d__%H-%M')
            githash = subprocess.check_output("git rev-parse --short HEAD", shell=True, universal_newlines=True)[:-1]

            git_path = subprocess.check_output("git rev-parse --show-toplevel", shell=True, universal_newlines=True)[:-1]
            entry_point = str(Path(main_file).resolve().relative_to(git_path))

            save_dir = Path(git_path) / 'runs' / args['game'] / commit_date
            save_dir.mkdir(parents=True, exist_ok=True)
        except:
            commit_date = 'untracked'
            githash = 'untracked'
            entry_point = str(Path(main_file).resolve())
        
            save_dir = (Path('/mnt/default/runs') if args["using_server"] else Path('runs')) / args['game'] / commit_date
            save_dir.mkdir(parents=True, exist_ok=True)

        c = 1
        while (save_dir/f'{c}.data').is_file():
            c += 1
        self.data_path = save_dir/f'{c}.data'

        data = f'game:\t\t\t{args["game"]}\n\ncommit_date:\t\t\t{commit_date}\ngithash:\t\t\t{githash}\nentry point:\t\t\t{entry_point}\n\n\n' + '\n'.join([f'{key}:\t\t\t{args[key]}' for key in sorted(args.keys()) if key != 'game'])
        Path(self.data_path).write_text(data)
        fname = '_'.join(k+"_"+str(v) for k,v in args.items() if  k != "using_server")
        self.npy_path = save_dir / f'{c}-{fname}.npy'  # or just save_dir/f'{c}.npy'
        self.pl_path = save_dir / f'{c}-{fname}.pl'


    def save(self, hoc):
        # np.save('probs'+npy_path, hoc.probs_hist)
        np.save(self.npy_path, hoc.history)
        with open(self.pl_path, 'wb') as dilldumpweight:
            dill.dump({
                'terminations': [ (o.uid, o.termination.model) for o in hoc.options if hasattr(o.termination,'model')],
                'options_weights':[ (o.uid, o.policy.pi) for o in hoc.options if hasattr(o.policy,'pi')],
                'options_q': [ (o.uid, o.policy.q) for o in hoc.options if hasattr(o.policy, 'q')],
                'options_v': [ (o.uid, o.policy.v) for o in hoc.options if hasattr(o.policy, 'v')]
                }, dilldumpweight)

        print(f'\n\nThe following files have been saved:\n  {self.data_path}\n  {self.npy_path}\n  {self.pl_path}\n\n')


    def __call__(self, hoc, run, episode, time):
        args = self.args

        print('*'*20 + (f'New goal: {hoc.env.goal}' if hasattr(hoc.env, 'goal') else ''))
        print(f'Run {run} episode {episode} steps {time} discounted-cumreward {hoc.options[0].reward}')
        # print('moving average ', moving_average(hoc.history[run, episode-499:episode+1, 0, 0])) #for fourrooms
        # print('moving average ', moving_average(np.sum(hoc.history[run, episode-99:episode+1, -2:, 1],axis= 1) ,100)) #for MDP
        if episode == args['nepisodes']-1: ### Save at the end of each run (overwrite if exists)
            self.save(hoc)