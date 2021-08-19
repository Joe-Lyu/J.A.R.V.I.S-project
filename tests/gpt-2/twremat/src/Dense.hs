module Dense where

type Dense = [Int]

between :: Dense -> Dense -> Dense
between (a:as) (b:bs)
  | a + 1 < b = [div (a + b) 2]
  | a     < b = a : after as
  | a    == b = a : between as bs

before :: Dense -> Dense
before [] = error "before []"
before (a:as) = [a - 1]

after :: Dense -> Dense
after [] = [2^20]
after (a:as) = [a + 1]
