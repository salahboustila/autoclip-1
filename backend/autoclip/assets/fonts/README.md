# Bundled fonts

All fonts here ship with AutoClip so nothing is fetched at runtime.

| File | Family | Copyright | License |
| --- | --- | --- | --- |
| `Anton-Regular.ttf` | Anton | The Anton Project Authors | SIL OFL 1.1 |
| `Inter-Variable.ttf` | Inter | The Inter Project Authors | SIL OFL 1.1 |
| `Montserrat-ExtraBold.ttf` | Montserrat ExtraBold | 2011 The Montserrat Project Authors (https://github.com/JulietaUla/Montserrat) | SIL OFL 1.1 |
| `emoji/NotoColorEmoji.ttf` | Noto Color Emoji | 2013-2017 Google Inc. (https://github.com/googlefonts/noto-emoji) | SIL OFL 1.1 |

The full license text is in [OFL.txt](OFL.txt).

Noto Color Emoji lives in `emoji/` on purpose: the caption renderer copies every
font at this level into each ffmpeg workspace for libass, and the emoji font is
10 MB that libass never uses. It is read only by the Pillow hook-title renderer.
