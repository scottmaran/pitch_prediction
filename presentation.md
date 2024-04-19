
## Data Preprocessing

- Remove FA, bc not sure if FF or FT (and only 204 samples)
- Decision: FT (two seamer) and SI (sinker) usually grouped together. Same speed as FF (four-seamer fastball), but:
    FT: more horizontal movement, "tailing"
    SI: more vertical drop
    - May have enough data where we'll keep separate
    - Fangraphs example groups together (https://community.fangraphs.com/no-pitch-is-an-island-pitch-prediction-with-sequence-to-sequence-deep-learning/)

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

## Examples

I got the average throw rates for each pitch type.

I then looked to see what kind of deviations from these averages existed.

For example, FF = 35.4%. Pitcher 547973 has FF=83.9% over 877 pitchers. He only ever throws two kinds of pitches. 

## Prior work

Paper by Glenn Sidle and Hien Tran (2015)
Overview:
    - RAN AN INDIVIDUAL MODEL FOR EACH PITCHER
    - Used data from three seasons (2013-2015) to predict pitch type, achieved 66.62% accuracy using LDA, SVM, RF
    - binary prediction (fastball vs non-fastball) usueally around 70% accuracy
    - Seven pitch categories and those that had type confidence greater than 80%
    fastball (FF, 1), cutter (CT, 2), sinker (SI, 3), slider (SL, 4), curveball (CU, 5), changeup (CH, 6), and knuckleball (KN, 7)
    - only used pitchers with at least 500 pitches in a season, so 287 unique pitchers (150 starters vs 137 relievers)
    - average pitcher had 81 features
    - Features: pretty much what I was thinking. Get game state, pitchers prior tendencies, and batter's tendencies
Model:
    - 66.62% accuracte with RF
    - gives prediction accuracy for each pitch count 
    - looked at correlations - the higher the HHI (how often throw the same pitch), the more accurate the model (R^2=.77)
    - gave feature importance 

IIE Paper
Overview:
    - Also individual models for pitches
    - Did some weird/interesting sequence manipulation to have more samples
    - LSTM models and XGBoost

## Decisions

### Model for each pitcher or not
    - Having a single model for each pitcher could better capture each pitcher's unique style
    - But, doesn't generalize well to new pitchers

### Through out low confidence, or weight each pitch

### Prior pitches thrown in previous matchup
    - Can be useful, but sparse and misleading bc of small sample size (as asserted by Ryan Plunkett).
    - Ryan tries to create more stable features by using some similarity measure
    - Given only have one season, estimates wouldn't be as stable

### Categorical data
    - Are balls, outs, strikes categorical? Ordinal, but if treat at categorical, then 2 outs and 0 outs difference equal to 1 outs and 0 outs, which not true

### Data splits
    - time series so need to split like that

## Models
- One big XGBoost for all
- Individual XGBoost for each 
- One big Transformer for all
- Transformer for each 
(if time, LSTM for each)