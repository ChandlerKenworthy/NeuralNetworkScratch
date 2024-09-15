# Makes some synthetically generated data to "train" the model on
# in this case it is attributes of different people and we are trying to predict their sex
import numpy as np
import pandas as pd

N_SAMPLES = 1000

# Approximately 50% male and 50% female
isMale = np.random.randint(0, 2, size=N_SAMPLES)

def make_synthetic_from_normal(mu, std):
    data = np.where(
        isMale == 1,
        np.random.normal(loc=mu[0], scale=std[0], size=1),
        np.random.normal(loc=mu[1], scale=std[1], size=1)
    )
    # Add some noisy perturbations
    noise = np.random.normal(loc=0, scale=np.mean(std)/4, size=data.shape)
    return data + noise

# Height [cm]
heights = make_synthetic_from_normal([175, 155], [25, 34])
# Weights [kg]
weights = make_synthetic_from_normal([75, 60], [9, 8])
# Shoe size
shoe_size = make_synthetic_from_normal([8.5, 2.0], [4.0, 3.0])
# Hair colour flag
has_brown_hair = np.random.randint(0, 2, size=N_SAMPLES)

# Combine into a dataframe and save it
df = pd.DataFrame({
    "is_male": isMale,
    "height": heights,
    "weight": weights,
    "shoe_size": shoe_size,
    "has_brown_hair": has_brown_hair
})

df.to_csv('synthetic_sex_data.csv', index=False)
