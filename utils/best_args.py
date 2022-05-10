

##Best for seq-cifar10 and seq-mnist

best_args = {
      'seq-cifar10': {
            'crr': 
            {
                  300: 
                  {
                        'lr': 0.9,
                        'batch_size' : 64,
                        'batch_size_test' : 128,
                        'minibatch_size': 64,
                        'alpha': 0.75,
                        'beta': 1.75,
                        'gamma': 1.25,
                        'end_lr': 0.4,
                        'n_epochs': 50,
                        'seed': 2711,
                  },
                  600: 
                  {
                        'lr': 0.9,
                        'batch_size' : 64,
                        'batch_size_test' : 128,
                        'minibatch_size': 64,
                        'alpha': 0.75,
                        'beta': 1.75,
                        'gamma': 1.25,
                        'end_lr': 0.4,
                        'n_epochs': 50,
                        'seed': 2711,
                  },
            },
      },
      'seq-cifar100': {
            'crr': 
            {
                  300: 
                  {
                        'lr': 0.9,
                        'batch_size' : 64,
                        'batch_size_test' : 128,
                        'minibatch_size': 64,
                        'alpha': 0.75,
                        'beta': 1.75,
                        'gamma': 1.25,
                        'end_lr': 0.32,
                        'n_epochs': 50,
                        'seed': 2711,
                  },
                  600: 
                  {
                        'lr': 0.9,
                        'batch_size' : 64,
                        'batch_size_test' : 128,
                        'minibatch_size': 64,
                        'alpha': 0.75,
                        'beta': 1.75,
                        'gamma': 1.25,
                        'end_lr': 0.32,
                        'n_epochs': 50,
                        'seed': 2711,
                  },
            },
      },
      'seq-core50': {
            'crr': 
            {
                  300: 
                  {
                        'lr': 0.9,
                        'batch_size' : 48,
                        'batch_size_test' : 48,
                        'minibatch_size': 48,
                        'alpha': 0.75,
                        'beta': 1.75,
                        'gamma': 1.25,
                        'end_lr': 0.36,
                        'n_epochs': 15,
                        'seed': 2711,
                  },
                  600: 
                  {
                        'lr': 0.9,
                        'batch_size' : 48,
                        'batch_size_test' : 48,
                        'minibatch_size': 48,
                        'alpha': 0.75,
                        'beta': 1.75,
                        'gamma': 1.25,
                        'end_lr': 0.36,
                        'n_epochs': 15,
                        'seed': 2711,
                  },
            },
      },
},