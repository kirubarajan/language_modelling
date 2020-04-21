# LSTM Language Model and Decoding Strategies
This code repository accompanies my blog post on different [decoding strategies for language models](https://kirubarajan.com/blog/decoding).

## Example Generations

Trained on Barack Obama speeches:

> I am proposing with an advantage over commerce budget. — (applause) the middle of commerce, way together more of each other people’s it. In the chance that the international issue, freedom we have never has allowed the other way, or share from footing or denied coverage for the work of Democrats and Republican administrations isn’t (Applause.) Now, none of this can happen unless we’re their own rules that progress on so tied long still blind you should make Wall good example. (Applause.) For unemployment to pull all we should leave just like us — (applause)


> I am proposing tax rates as true build today. not less. But this chamber can point to work together on a fair shot. billion better we’re becoming when people will earn them out. where we built it before. in to the rest of the world. It makes no — these Americans have your it because they may have passed every American people. (Applause.) Second, a vote. would call creating country we will support for the new economy, we have one going through all expectations.

## Usage
1. Install PyTorch using `pip install torch torchvision`.
2. Train the LSTM model using `python train.py`.
