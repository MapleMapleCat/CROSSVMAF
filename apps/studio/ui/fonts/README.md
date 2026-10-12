# Fonts

Inter and JetBrains Mono are the existing Studio UI fonts. The PDF exporter
embeds Inter Regular / SemiBold and Noto Sans SC Regular as a CJK fallback.
PDF files carry compressed font programs and Unicode maps, so recipients do
not need to install fonts and can search or copy the report text.

`NotoSansSC-Regular.ttf` is the weight-400 static instance of Google Fonts'
`ofl/notosanssc/NotoSansSC[wght].ttf` at commit
`2894aab31764f10f29c421bdfd2340d3b382d384`:
https://github.com/google/fonts/tree/2894aab31764f10f29c421bdfd2340d3b382d384/ofl/notosanssc

It was generated with fontTools 4.63.0:

```python
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
font = TTFont("NotoSansSC-variable.ttf", recalcTimestamp=False)
instantiateVariableFont(font, {"wght": 400}, inplace=True, updateFontNames=True)
font.save("NotoSansSC-Regular.ttf")
```

Static asset SHA-256:
`eeb06b8a64fd04a2744d95579db1571b51027cda61ed78c62e4b730791525461`.

The upstream OFL notice is preserved in `OFL-NotoSansSC.txt` and staged in
Studio's license directory. FontTools is needed only to reproduce this asset;
normal builds use the Python standard library to embed the checked-in fonts.
The bundled fallback covers Chinese and many other scripts; unsupported
glyphs display as `?` while the PDF Unicode map preserves the original text.
