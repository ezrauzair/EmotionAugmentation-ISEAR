import nlpaug.augmenter.word as naw
import pandas as pd

# Initialize the synonym augmenter
aug = naw.SynonymAug(aug_src='wordnet')

# Load the dataset
df = pd.read_excel('/content/ISEAR.xlsx')

# Create an empty DataFrame to store augmented data
df_augmented = pd.DataFrame(columns=['content', 'labels'])

# Loop through each label (0 to 6)
for label in range(7):
    # Filter rows where the label equals the current label in the loop
    df_label = df[df['labels'] == label]

    # Augment the 'content' column for the filtered rows
    df_label['content'] = df_label['content'].apply(lambda x: aug.augment(x)[0] if aug.augment(x) else x)

    # Append the augmented data to the new DataFrame
    df_augmented = pd.concat([df_augmented, df_label], ignore_index=True)

# Combine the original dataset with the augmented dataset
df_combined = pd.concat([df, df_augmented], ignore_index=True)

# Display the final dataset with original and augmented data
print(df_combined)
