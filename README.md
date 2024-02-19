# heating-planner

## Why

Gathering open source data about global heating's geographic trends and merge them into a usable informative digest.
Example here : https://heating-planner-yzaj3jxgfwpquuhq3ofx6t.streamlit.app/ 

## How to ?

Collect maps from <https://interactive-atlas.ipcc.ch> by screen-shoting them (maybe later one will provide a handier way to collect data). Collect screenshots + metadata for each screenshot. See example:

|    | variable   | season    |   scale_mini |   scale_maxi | unit      | init_fname                                |
|---:|:-----------|:----------|-------------:|-------------:|:----------|:------------------------------------------|
| 19 | TXx        | Automne   |            5 |           38 | degrees C | Capture d’écran 2023-08-01 à 15.52.09.png |
| 29 | RX5day     | Printemps |           15 |          180 | mm        | Capture d’écran 2023-08-01 à 16.14.17.png |
| 16 | TXx        | Hiver     |          -15 |           20 | degrees C | Capture d’écran 2023-08-01 à 15.48.57.png |

Where

- `variable` is the id of the recorded variable
- `season` is the recorded season
- `scale_mini` is the lowerbound of the colorbar used
- `scale_maxi` is the upperbound of the colorbar used
- `unit` is the unit associated to the variable
- `init_fname` is the initial name of your screenshot
- `init_folder` is the initial folder of your screenshot
