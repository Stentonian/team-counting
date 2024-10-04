# Combinatorics for hackathons

There are $N$ people attending a hackathon. The rules of the hackathon dictate that hackers can get into teams of size $1,2,3,...,t$. How many teams are going to form if hackers choose their teams randomly? Is the distribution random? Or is it more likely that a certain number of teams will form?

## Problem statement

We have $N$ people, and teams of size $1,2,3,...,t$

We want the probability of $k$ teams occurring,
assuming that people form teams of any allowable size at random. More specifically, we want a function $P(k)$ that returns the probability of $k$ teams forming at random.

This problem is similar to how Boltzman defined entropy using statistical mechanics.

Example counting procedure:
for $N=10$ and only teams of size 1 & 2 ($t=2$)
1. first, put everyone into a size-1 team (singles); there are 10 teams then, and only 1 possible configuration
2. next, have 1 team of 2 and the rest singles; there are 9 teams then, and $10 \choose 2$ possible configurations
3. ...
So we have a map of `num_teams -> num_combinations`

## Details

### Suppose $t=2$

The number of ways of pairing $m=2n$ people off is ([source](https://math.stackexchange.com/questions/1234696/number-of-ways-you-can-form-pairs-with-a-group-of-people-when-certain-people-can))

$$\alpha _2(n)= \frac{(2n)!}{n!2^n}$$

Given $N$ people, suppose $m\le N$ of those people are going to pair off (the rest will be in teams by themselves) then the number of combinations is

$$\beta _2(N,n)= {N \choose 2n}\alpha _2(n)$$

Note that the number of ways to arrange $n$ people into teams of size 1 is just 1, so once we have counted the number of pairs then we are done. Final probability function $P$ is:

$$P_2(N,k)= \frac{\beta _2(N,N-k)}{\sum _{n=1}^N \beta _2(N,n)}$$

where $k$ is the number of teams. Why $N-k$? This is one way to think about it: if we pick $m\le N$ people such that $m$ is even (say $m=2n$) and we pair off these people and leave the rest in teams by themselves, then the number of teams we have is $k=n+(N-2n)=N-n$ and the number of ways of forming this number of teams is $\beta _2(N,n)$. If we know $k$ but not $n$ then set $n=N-k$.

### Suppose $t=3$

We get similar counting functions to above:

$$\alpha _3(n)= \frac{(3n)!}{n!(3!)^n}$$

$$\beta _3(N,n)= {N \choose 3n}\alpha _3(n)$$

This time counting totals is more complicated, since teams of size 3, 2, & 1 are all possible. Imagine first picking a number $m_3\le N$ s.t. $3|m_3$, and making teams of size 3 with this number of people. Next, pick a number $m_2\le N-m_3$ and make teams of size 2 with this number of people. Finally, put the rest of the people into teams by themselves. The total number of teams is given by $k=n_3+n_2+(N-3n_3-2n_2)=N-2n_3-n_2$ where $in_i=m_i$. There may be more than one such tuple $(n_3,n_2)$ that gives a particular $k$, so we need to sum up the counts for each of these tuples to get our final probability:

$$P_3(N,k)=\frac{1}{S} \left[ \sum _\underset{k=N-2n_3-n_2}{m_3,m_2 \le N} \beta _3(N,n_3)\beta _2(N-m_3,n_2) \right]$$

where $S$ is the total sum of all the counts, i.e.

$$S=\sum _k \sum _\underset{k=N-2n_3-n_2}{m_3,m_2 \le N} \beta _3(N,n_3)\beta _2(N-m_3,n_2)$$

### General $t$

One can keep going with this game beyond $t=3$, but the math is a bit tedious to display in Github-compatible latex, so I'm not going to do it here. The code uses a recursive function to calculate for general $t$.

