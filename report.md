
# Goal 

Given all pitches from the 2011 MLB season, we want to predict the probability that the next pitch thrown will be one of multiple pitch types. There are 19 unique pitch types in the full dataset:

Fastballs:
    'FA', Fastball
    'FF', Four-Seam Fastball    - Carry (straight)
    'FT', Two-Seam Fastball     - Run 
    'SI', Sinker                - Drop 

    'FC', Fastball (cutter)     - Horizontal, opposite direction of two-seamer 
        - for RHP, right to left (like volleyball cut shot)

    'FS', Split-Fingered        - looks like fastball till drops
Others: - defined by movement 
    'SL', Slider                - sideways, same direction as cutter, more sweeping (i.e. see break happening from beginning to end)
    'CU', Curveball             - up and down 
    'CH', Changeup              - look like fastball but are slower (gets batter to swing too early), can have some fade (move right with righties)
    'KC', Knuckle-Curve
    'KN', Knucleball 
    'EP', Eephus                - rare, "slowball"
Other: 
    'UN', Unidentified
    'PO', Pitch out 
    'FO', Forkball
    'SC', Screwball

    'IN', Intentional Ball
    'AB', Automatic Ball 

## Exploratory Data Analysis

The distribution of these pitch types are:

pitch_type
FF    238541
SL    109756
SI     87740
FT     81056
CH     72641
CU     56379
FC     41702
FS     10503
KC      8490
KN      4450
IN      4058
PO       559
FO       329
FA       204
EP       134
SC       120
UN        17
AB         2

## Data PreProcessing

- I removed Fastballs (FA) from the dataset as (from my understanding) this could mean either a Two-Seam (FT) or Four-Seam (FF) Fastball and the number present in the dataset (204) was relatively insignificant (0.029%).
- There does not seem to be consensus on how important it is to keep the distinction between Two-Seam Fastball's (FT) and Sinker's (SI), as some people group them together and others don't. I have little baseball domain knowledge myself and thus chose to keep them separate. This was also decided because of each group's large sample size.

## Prior Work

## Decisions
### Model for each pitcher or not
One of the biggest decisions I had to make was if I wanted to use one single model, or develop a model for each individual pitcher. I was surprised to find the existing literature (DOMINATED) towards the latter. From the LSTM paper:

> "Because the available pitch types differ from pitchers, this work
    trained individual models for each pitcher rather than having
    a unified model. We believe that this approach is much close
    to the practical situation. The output categories (i.e. the pitch
    type) of each model vary from 3 to 9" - [Yu, Chang, and Cheng](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9859411)

A similar sentiment is echoed in the Random Forest paper.

This is surprising to me for many reasons.
1) From the above quote, if each individual model can't predict all of the possible categories, this means that the model can never predict a pitch that the pitcher has never previously thrown.
2) Splitting into multiple models has two huge drawbacks - you significantly reduce the sample size available, and also weaken the potential gloabl properties that a single model would have access to across pitchers
3) If you want to predict pitches on unseen PITCHERS, you need to build a whole new models
4) If you want to predict pitches on pitchers with a really low sample size, ... ¯\\_(ツ)_/¯

A global model eliminates all of these concerns

### Through out low confidence, or weight each pitch

### Prior pitches thrown in previous matchup

### Categorical data

### Data splits

## Models

## Results

## Future Work

- Use previous seasons to get better estimates of historical tendencies