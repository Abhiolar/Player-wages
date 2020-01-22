## FOOTBALL TRANSFER MARKET PROBLEM
![Screenshot 2020-01-19 at 18 48 09](https://user-images.githubusercontent.com/53583240/72686826-63d9c680-3af0-11ea-9ffa-8b0f5801e41c.png)

 
 ## BUSINESS CASE 
 
 My goal is to build  a machine learning model that can predict the wages of football players that will be beneficial to premier league teams when they are signing new players based on their performance in the previous season (2018/2019) and get some insights and analysis into what type if players clubs would be interested in, thereby making a decision on whether to buy them based on certain criteria

I aim to build a machine learning model using numerous regression techniques and getting the best predictor model  that is ideal for your club as a premier league director of football, manager or head of scout.

Features of the dataset include the player's weekly wages, transfer market value, total minutes played in the season, age of player, the amount of goals and assists, amount of red and yellow cards,  amount of clean sheets, rank in goals/club.
 
The business problem is the fact that a lot of premier league clubs overpay in transfer fees and player salaries and it has led to increase in sports tv subscriptions such as SKY AND BT SPORT, merchandise and stadium ticket prices. This problem seems to getting worse with each passing year.

Some premier league clubs have approached my data science company and tasked me with finding how they can curb this trend.

My aim is to develop machine learning models that can make sure that these clubs are not overpaying these players and they are not overpaying in the transfer market either.'
 


### PROJECT DELIVERABLES
1. Final Notebook that details the whole entire project
2. Working regression model
3. 5 Business cases examples that delves into the analysis of getting the best value of player for the right amount of money.


### WHY IS THIS IS A PROBLEM AND IS IT RELEVANT?
Wage Inflation is a big problem in World football as evidenced by this article written by FourFourTWO: https://www.fourfourtwo.com/features/overpaid-and-underplayed-what-mesut-ozil-and-alexis-sanchez-tell-us-about-footballs-market. 
Alexis Sanchez and Mesut Ozil are two of the most popular footballers in the world. Based on their performances in the years gone by, these two superstars were the driving forces behind most of Arsenal's attacking play but Manchester United came in 2018 and got the Chilean, signing a £350,000 weekly wage contract up from £130,000, despite poor performances at Arsenal and signs of decline in ability.

Arsenal got desperate after the departure of Alexis Sanchez and gave their other talisman Mesut Ozil a £300,000 weekly wage contract making him the highest paid player in the history of the club,but the coach Unai Emery barely played the Attacking Midfielder.

So the problem is you have two players, one doesn't give good value for money and the other doesn't have the chance to give good value for money.

This problem is relevant because fans suffer for the rising wages of these overpaid players with rise in TV subscriptions, stadium and Merchandise. But most of all, clubs record decrease in profits at the end of the year.
![10858118-6797523-image-a-24_1552347400993](https://user-images.githubusercontent.com/53583240/72687280-31ca6380-3af4-11ea-9772-52712ac03307.jpg)




### 5 BUSINESS QUESTIONS WE NEED TO ANSWER
This set of business questions will take on a hypothetical scenario of the data science company representing some of the top clubs, for example Manchester United, Liverpool, Chelsea, Real Madrid etc...

#### 1.Chelsea has approached the business, as they are in the need of top 10 defenders in the premier league that have high goal involvement ratio per 90 minutes and have played over 2000 minutes in 2018/2019 season
The reason why Chelsea FC have gone for 2000 minutes(at least 22 games) is because they want a player that is not injury prone, plays consistently and can contribute goals from the defense either by making or scoring the goals.

Insights - Jose Holebas has the best goal involvment ratio per 90 minutes, the Watford defender is 35 years with a market value of £2.5M and £39,000 in weekly wages, even thou Andy Robertson is quite young at 25 years, his weekly wage is £135,000. We suggest Ryan Sessegnon, who is third best on the list- very young at 19 years of age and commands a lower weekly wage of £38,000.

![fulham-v-brentford-sky-bet-championship-craven-cottage-390x285](https://user-images.githubusercontent.com/53583240/72687537-91c20980-3af6-11ea-8c38-c7abc2b8df45.jpg)


#### 2. Manchester has approached the data science team, they are looking for a player (forward/midfielder) in the top leagues that consistently converted penalties in the 2018/2019 season?
Why is this a problem for Manchester United? Manchester United were awarded the most penalties in the 2018/2019 season (12 in total..most in the league), they scored 9 and had 3 penalties saved. The manager has deemed this "not good enough" -solution is to find a player in the top leagues that amassed more than 2500 minutes of football and has a goal involvement ratio greater than 0.5. They have stated money is not an issue as they only want the best.


Insights - I would recommend Bruno Fernandes of Sporting CP to Manchester United as he has the sixth best penalty conversion rate, he is 25 years of age and has a higher goal involvement ratio than Simon Gustafson of Utrecht in fourth and he is also cheaper in terms of weekly wages and 6 years younger than Dusan Tudic in first place. Although Nicolas Pepe is younger, he commands a higher weekly wage at £97,000 and lower goal_involvment ratio at 0.89.


![22855448-0-image-a-82_1577830219954](https://user-images.githubusercontent.com/53583240/72687656-e9ad4000-3af7-11ea-87a5-74f9cb1614a5.jpg)

#### 3.Atletico Madrid club in the spanish league has approached the data science team- with their tight transfer budget, they asked us to find and suggest a forward in premier league that has the highest ratio of goals/market value

This is a classic case of return on investment, and it is a very valid proposal- essentially saying----get us the striker that is worth his weight in gold.

Insights - Brighton striker Glenn Murray, last season's performances in terms of goals compared to his market value is the best in the whole of the premier league but at 36, he might not represent a good return on investment in the long term in terms of resale value.
We as Data Scientists suggest Callum Wilson, although he is 5 years older than Richarlison who is the youngest in the group. He has more more goals and assists than any player in the top 10, difference of 0.20 goal involvment ratio than second on the list Solomon Random. He also comes in tied second youngest on the list, which means more years to play at the top level before retirement.

![179489723 jpg gallery](https://user-images.githubusercontent.com/53583240/72687933-a4d6d880-3afa-11ea-957b-922bd0abf600.jpg)


#### 4. Brighton and Southampton were 2 and 3 points from being relegated to the championship last season. They have tasked us with looking for a striker and a defender in the championship that can boost their performance in the upcoming season

It is necessary these two teams get some good reinforcements that can prevent the struggles of last season. The first point of call is getting a striker to put the ball in the back of the net and defender to stop leaking of goals. These small teams don't have the transfer budgets of big clubs hence they have specifically asked for English championship top performers.

Insights  - George Baldock, John Egan and Enda Stevens are all Sheffield United defenders and they had the least amount of goals conceded for every 90 minutes they were on the pitch. We suggest a bid for George Baldock would be the best option, he is 26 and cheapest out of the first four options in terms of weekly wages at £25,000 and he is worth a market_value of £4.2M

![george_baldock_of_sheffield_united_during_the_premier_league_mat_1305598](https://user-images.githubusercontent.com/53583240/72687769-29285c00-3af9-11ea-9066-a4c2d6bb92f5.jpg)

Insights  - Norwich striker Teemu Pukki is the highest performing forward in the championship ahead of Neal Maupay and big name forward "Tammy Abraham", We suggest Neal Maupay as an ideal buy, one year older than Tammy Abraham, six years and ten years younger than Dwight Gayle and Billy Sharp respectively with a weekly wage demand of £40,000 which is £37,000 cheaper than Tammy Abraham.

![neal-maupay-fc-brentford-1565030618-24322](https://user-images.githubusercontent.com/53583240/72687863-00ed2d00-3afa-11ea-8a00-f3d03e575ad1.jpg)



#### 5.AS ROMA, an Italian football club approached the data science team and would very much like to plan for the future in their goalkeeping department, having shipped in 48 goals last season in the SERIE A. They plan on delving into the european transfer market to get the best goalkeeper under the age of 25.

Insights - I would recommed FC Lille goalkeeper Mike Maignan to AS Roma, He is exactly the kind of goalkeeper they are looking for, with his wages at £28,000 and age at 24. He has the lowest goals conceded for every 90 minutes.

![mike-maignan-losc-lille-1539597868-18247](https://user-images.githubusercontent.com/53583240/72687896-4f9ac700-3afa-11ea-82ee-0e8ba4e7de70.jpg)


## REGRESSION MODELLING

For the initial ordinary least squares regression modelling:
No signs of multicollinearity although the adjusted R-squared is a very low value, we get a better more robust model which enables each feature in the model to be truly independent and each variable can be varied with others kept constant.

There is still a lot of skew and kurtosis in the distribution of the model results but this might be atrributed to the magnitude of each value not being on the same scale.More pre-processing is definitely needed.
The positive skewness indicates a long tail to the right and postive kurtosis value more than 3 points to having too great a peak.
![Screenshot 2020-01-19 at 20 33 42](https://user-images.githubusercontent.com/53583240/72687998-4827ed80-3afb-11ea-8ecb-db581974feeb.png)

![Screenshot 2020-01-19 at 20 41 08](https://user-images.githubusercontent.com/53583240/72688107-1c593780-3afc-11ea-8150-2dd36469ce72.png)


## BASELINE LINEAR REGRESSION AND POLY-REGRESSION RESULTS
![Screenshot 2020-01-19 at 20 48 59](https://user-images.githubusercontent.com/53583240/72688250-35161d00-3afd-11ea-8d36-56f112684846.png)

As you can see from above, the model fit improved with r-squared at 0.67 for the test data and 0.68 for the training data. The MSE also decreases when compared to the first two models in the notebook, and the model is genreralizable and more robust, as the difference in the MSE bewteen the train and test data is low

Accounting for interactions in a model is quite important, what you are essentially doing is transforming a variable role in a linear regression based on another one, using PolyNomial features allows both interactions and polynomial expansions. The Polynomial regression results for the train and test data is a big improvement on the baseline model with a lower MSE, higher r squared for the better model fit and most importantly generalizable to test data

## XGBREGRESSOR RESULTS
![Screenshot 2020-01-19 at 20 54 21](https://user-images.githubusercontent.com/53583240/72688353-28de8f80-3afe-11ea-9259-ed0ad426bd91.png)

The xgb regressor is the best performing model with a r-squared value of 0.76 for the test and MSE value of 1.0e08. This ensemble method works by sequentially adding predictors to an ensemble and each one correcting its predecessor. It works to improve the weak learners in the group of learners, it tries to fit a new predictor to the residual errors made by the previous predictor.


## MODEL TEST ON NEW DATA

![Screenshot 2020-01-19 at 21 07 11](https://user-images.githubusercontent.com/53583240/72688500-cd150600-3aff-11ea-9ca1-24c7f5ba65bb.png)

The results clearly shows that the model can generalise on new test data, for example Bruno Fernandes, best player for sporting lisbon earned £24,000 in the portugese league whereas his true worth in my model prediction is roughly £60,000. This is based on his incredible performances in the previous season, he recently signed a new contract for his club (sporting Lisbon) for £65,000. My model is not too far away from his true wages






