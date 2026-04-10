General
𝑁
N-dimensional discrete symbolic complexity

Let the system state at time
𝑡
t be a lattice field:

𝑋
𝑡
:
Ω
⊂
𝑍
𝑁
→
{
0
,
1
,
…
,
𝑘
−
1
}
X
t
​

:Ω⊂Z
N
→{0,1,…,k−1}

where:

𝑁
N = spatial dimension,
𝑘
k = number of discrete states.

Define the full complexity functional as:

𝐶
𝑁
=
𝑊
𝐻
(
𝐻
ˉ
,
𝜎
𝐻
)
⋅
(
𝑊
𝑂
𝑃
,
𝑠
(
𝑁
)

- 𝑊
  𝑂
  𝑃
  ,
  𝑡
  )
  ⋅
  𝑊
  𝑇
  (
  𝑇
  𝐶
  ˉ
  )
  ⋅
  𝑊
  𝐺
  (
  𝑅
  )
  ⋅
  𝑊
  𝐷
  C
  N
  ​

=W
H
​

(
H
ˉ
,σ
H
​

)⋅(W
OP,s
(N)
​

+W
OP,t
​

)⋅W
T
​

(
TC
ˉ
)⋅W
G
​

(R)⋅W
D
​

    ​

where
𝑊
𝐷
W
D
​

is optional (fractal / dimensional occupancy).

1. Entropy term

For each time
𝑡
t:

𝐻
(
𝑡
)
=
−
∑
𝑖
=
1
𝑘
𝑝
𝑖
(
𝑡
)
log
⁡
2
𝑝
𝑖
(
𝑡
)
H(t)=−
i=1
∑
k
​

p
i
​

(t)log
2
​

p
i
​

(t)

Normalize:

𝐻
~
(
𝑡
)
=
𝐻
(
𝑡
)
log
⁡
2
𝑘
H
~
(t)=
log
2
​

k
H(t)
​

Then:

𝐻
ˉ
=
1
𝑇
∑
𝑡
𝐻
~
(
𝑡
)
,
𝜎
𝐻
=
1
𝑇
∑
𝑡
(
𝐻
~
(
𝑡
)
−
𝐻
ˉ
)
2
H
ˉ
=
T
1
​

t
∑
​

H
~
(t),σ
H
​

=
T
1
​

t
∑
​

(
H
~
(t)−
H
ˉ
)
2
​

Weight:

𝑊
𝐻
=
tanh
⁡
(
50
𝐻
ˉ
)
tanh
⁡
(
50
(
1
−
𝐻
ˉ
)
)
[
1

- exp
  ⁡
  (
  −
  (
  𝜎
  𝐻
  −
  0.012
  )
  2
  2
  (
  0.008
  )
  2
  )
  ]
  W
  H
  ​

=tanh(50
H
ˉ
)tanh(50(1−
H
ˉ
))[1+exp(−
2(0.008)
2
(σ
H
​

−0.012)
2
​

)] 2. Spatial opacity in
𝑁
N dimensions

Let:

𝐿
𝑡
(
𝑥
)
L
t
​

(x) = local neighborhood patch around site
𝑥
x,
𝐺
𝑡
G
t
​

= coarse global macrostate (density, magnetization, etc.).

Then:

𝑂
𝑃
𝑢
𝑝
(
𝑁
)
=
𝐻
(
𝐺
∣
𝐿
)
log
⁡
2
𝑛
𝑏
𝑖
𝑛
𝑠
OP
up
(N)
​

=
log
2
​

n
bins
​

H(G∣L)
​

𝑂
𝑃
𝑑
𝑜
𝑤
𝑛
(
𝑁
)
=
𝐻
(
𝐿
∣
𝐺
)
log
⁡
2
∣
𝐿
∣
OP
down
(N)
​

=
log
2
​

∣L∣
H(L∣G)
​

Weight:

𝑊
𝑂
𝑃
,
𝑠
(
𝑁
)
=
exp
⁡
(
−
(
𝑂
𝑃
𝑢
𝑝
(
𝑁
)
−
0.14
)
2
2
(
0.10
)
2
)
⋅
exp
⁡
(
−
(
𝑂
𝑃
𝑑
𝑜
𝑤
𝑛
(
𝑁
)
−
0.97
)
2
2
(
0.05
)
2
)
W
OP,s
(N)
​

=exp(−
2(0.10)
2
(OP
up
(N)
​

−0.14)
2
​

)⋅exp(−
2(0.05)
2
(OP
down
(N)
​

−0.97)
2
​

)

This is the core of your “hierarchical opacity” idea.

3. Temporal opacity

Already dimension-independent:

𝑀
𝐼
1
=
𝐼
(
𝑋
𝑡
;
𝑋
𝑡

- 1
  )
  𝐻
  (
  𝑋
  𝑡
  )
  MI
  1
  ​

=
H(X
t
​

)
I(X
t
​

;X
t+1
​

)
​

𝑑
𝑒
𝑐
𝑎
𝑦
=
𝑀
𝐼
1
−
𝑀
𝐼
𝑙
𝑎
𝑔
decay=MI
1
​

−MI
lag
​

Weight:

𝑊
𝑂
𝑃
,
𝑡
=
tanh
⁡
(
10

 
𝑀
𝐼
1
)
tanh
⁡
(
10
(
1
−
𝑀
𝐼
1
)
)
tanh
⁡
(
10

 
𝑑
𝑒
𝑐
𝑎
𝑦
)
W
OP,t
​

=tanh(10MI
1
​

)tanh(10(1−MI
1
​

))tanh(10decay) 4. Temporal compression

Per site:

𝑇
𝐶
𝑖
=
1
−
1

- flips
  𝑖
  𝑇
  TC
  i
  ​

=1−
T
1+flips
i
​

    ​

Average:

𝑇
𝐶
ˉ
=
1
∣
Ω
∣
∑
𝑖
𝑇
𝐶
𝑖
TC
ˉ
=
∣Ω∣
1
​

i
∑
​

TC
i
​

Weight:

𝑊
𝑇
=
max
⁡
[
𝐺
(
𝑇
𝐶
ˉ
;
0.58
,
0.08
)
,
𝐺
(
𝑇
𝐶
ˉ
;
0.73
,
0.08
)
,
𝐺
(
𝑇
𝐶
ˉ
;
0.90
,
0.05
)
]
W
T
​

=max[G(
TC
ˉ
;0.58,0.08),G(
TC
ˉ
;0.73,0.08),G(
TC
ˉ
;0.90,0.05)] 5. Compressibility

Flatten the spacetime tensor and compress:

# 𝑅

compressed size
raw size
R=
raw size
compressed size
​

Weight:

𝑊
𝐺
=
exp
⁡
(
−
(
𝑅
−
0.10
)
2
2
(
0.05
)
2
)
W
G
​

=exp(−
2(0.05)
2
(R−0.10)
2
​

)
