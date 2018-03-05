# Scrying Pen

This pen's ink stretches backwards into the past and forwards into possible futures. The two sides make a strange loop: the future ink influences how you draw, which in turn becomes the new "past" ink influencing further future ink.

Put another way: this is a realtime implementation of [SketchRNN](https://arxiv.org/abs/1704.03477) which predicts future strokes while you draw.

[Play with it here!](http://andymatuschak.org/scrying-pen) (You'll need to use Chrome for now.)

![video of toy in action](https://andymatuschak.org/scrying-pen/images/hand.gif)

Let it guide your hand, or not, while you draw. Enjoy the gentle interplay between your volition and the machine's.

## Background

[I'll be writing more about the background of this project before talking about it broadly; a brief sketch follows:]

I'm excited about applications of machine learning to human augmentation—as opposed to automating tedious work or solving number-crunching problems. I believe effective media for human augmentation require feedback loops tight enough to work at the speed of thought. All that is to say: I'm interested in what happens when we take complex operations and make them interactive in real-time.

## Thanks

I'm grateful to David Ha, Douglas Eck, and their team for their great work with SketchRNN. The paper's wonderful to read; the models from Quick, Draw! are a great resource to the community; the demo implementations were a very helpful reference.

I'm also grateful to Michael Nielsen, Robert Ochshorn, and M Eifler for useful conversations about this project.

Thanks also to the [OpenAI Hackathon](https://blog.openai.com/hackathon/) for providing a nice venue for polishing the last bits of this project up.
