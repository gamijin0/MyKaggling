import xlearn as xl

# Training task
ffm_model = xl.create_ffm()  # Use factorization machine
ffm_model.setTrain("data/train_for_xlearn.csv")  # Training data

# param:
#  0. Binary task
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  4. evaluation metric: rmse
param = {'task': 'reg', 'lr': 0.2,
         'lambda': 0.0002, 'metric': 'rmse', 'epoch': 1000}

# Use cross-validation
ffm_model.fit(param, "data/model.out")
