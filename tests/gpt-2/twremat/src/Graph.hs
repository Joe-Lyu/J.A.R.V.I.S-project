module Graph where

import           Control.Monad.State.Strict
import           Data.Foldable
import           Data.IntMap (IntMap)
import qualified Data.IntMap as IM
import           Data.IntSet (IntSet)
import qualified Data.IntSet as IS
import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.Set (Set)
import qualified Data.Set as Set
import           Text.Printf

type Node = Int
type Context a = (IntSet, a, IntSet)
newtype Gr a = Gr (IntMap (Context a))

instance Functor Gr where
  fmap = nmap

instance Show a => Show (Gr a) where
  showsPrec d g = showParen (d > 10) $ showString "mkGraph " . showsPrec 11 (labNodes g) . showString " " . showsPrec 11 (edges g)

mkGraph :: [(Node, a)] -> [(Node, Node)] -> Gr a
mkGraph nodes edges = Gr (IM.fromList [(v, ctx v a) | (v, a) <- nodes])
  where
    ctx v a = (IM.findWithDefault IS.empty v bwd, a, IM.findWithDefault IS.empty v fwd)
    fwd = IM.fromListWith (<>) [(a, IS.singleton b) | (a, b) <- edges]
    bwd = IM.fromListWith (<>) [(b, IS.singleton a) | (a, b) <- edges]

mkUGraph :: [Node] -> [(Node, Node)] -> Gr ()
mkUGraph nodes edges = mkGraph (zip nodes (repeat ())) edges

labNodes :: Gr a -> [(Node, a)]
labNodes (Gr m) = l <$> IM.toList m
  where
    l (v, (p, a, s)) = (v, a)

nodes :: Gr a -> [Node]
nodes (Gr m) = IM.keys m

edges :: Gr a -> [(Node, Node)]
edges (Gr m) = foldMap go (IM.toList m)
  where
    go (v, (p, a, s)) = map ((,) v) (IS.toList s)

suc :: Gr a -> Node -> IntSet
suc (Gr m) v = case m IM.! v of
  (p, a, s) -> s

pre :: Gr a -> Node -> IntSet
pre (Gr m) v = case m IM.! v of
  (p, a, s) -> p

lab :: Gr a -> Node -> a
lab (Gr m) v = case m IM.! v of
  (p, a, s) -> a

labMaybe :: Gr a -> Node -> Maybe a
labMaybe (Gr m) v = case IM.lookup v m of
  Just (p, a, s) -> Just a
  Nothing -> Nothing

sucList :: Gr a -> Node -> [Node]
sucList g v = IS.toList (suc g v)

preList :: Gr a -> Node -> [Node]
preList g v = IS.toList (pre g v)

indeg :: Gr a -> Node -> Int
indeg gr v = IS.size (pre gr v)

outdeg :: Gr a -> Node -> Int
outdeg gr v = IS.size (suc gr v)

hasEdge :: Gr a -> (Node, Node) -> Bool
hasEdge (Gr m) (a,b) = case IM.lookup a m of
  Just (p, a, s) -> IS.member b s
  Nothing -> False

hasNode :: Gr a -> Node -> Bool
hasNode (Gr m) v = case IM.lookup v m of
  Just _ -> True
  Nothing -> False

delNode :: Node -> Gr a -> Gr a
delNode v (Gr m) = case IM.lookup v m of
  Just (p, a, s) -> Gr . foldr (.) id (clearSucc v <$> IS.toList p) . foldr (.) id (clearPred v <$> IS.toList s) . IM.delete v $ m
  Nothing -> Gr m
  where
    clearSucc v k m = IM.adjust (\(p, a, s) -> (p, a, IS.delete v s)) k m
    clearPred v k m = IM.adjust (\(p, a, s) -> (IS.delete v p, a, s)) k m

insEdges :: [(Node, Node)] -> Gr a -> Gr a
insEdges es (Gr m) = Gr . part1 . part2 $ m
  where
    adjs = IM.fromListWith (<>) [(a, IS.singleton b) | (a, b) <- es]
    adjp = IM.fromListWith (<>) [(b, IS.singleton a) | (a, b) <- es]
    part1 = foldr (.) id [IM.adjust (\(p, a, s) -> (p, a, s <> js)) i | (i, js) <- IM.toList adjs]
    part2 = foldr (.) id [IM.adjust (\(p, a, s) -> (p <> js, a, s)) i | (i, js) <- IM.toList adjp]

insNode :: (Node, a) -> Gr a -> Gr a
insNode (i, a) (Gr m) = Gr (IM.alter go i m)
  where
    go (Just (p1, a1, s1)) = Just (p1, a, s1)
    go Nothing             = Just (IS.empty, a, IS.empty)

(&) :: (IntSet, Node, a, IntSet) -> Gr a -> Gr a
(&) (p, i, a, s) = insEdges (ein ++ eout) . insNode (i, a)
  where
    ein = [(j, i) | j <- IS.toList p]
    eout = [(i, j) | j <- IS.toList s]

newNodes :: Int -> Gr a -> [Node]
newNodes n (Gr m) = case IM.findMax m of
  (x, _) -> [x+1..x+n]

-- XXX: Really slow.
order :: Gr a -> Int
order (Gr m) = IM.size m

gmap :: (Node -> Context a -> Context b) -> Gr a -> Gr b
gmap f (Gr m) = Gr (IM.mapWithKey f m)

nmap :: (a -> b) -> Gr a -> Gr b
nmap f (Gr m) = Gr (IM.map go m)
  where go (p, a, s) = (p, f a, s)

gfilter :: (Node -> Context a -> Bool) -> Gr a -> Gr a
gfilter f (Gr m) = Gr (IM.mapMaybeWithKey go m)
  where
    go i (p, a, s)
      | IS.member i keep = Just (IS.intersection p keep, a, IS.intersection s keep)
      | otherwise        = Nothing
    keep = IS.fromList [i | (i, ctx) <- IM.toList m, f i ctx]

labfilter :: (a -> Bool) -> Gr a -> Gr a
labfilter f g = gfilter (\i (p, a, s) -> f a) g

dfs :: Gr a -> [Node] -> [Node]
dfs g start = go start IS.empty
  where
    go [] visited = []
    go (x:xs) visited
      | IS.member x visited = go xs visited
      | otherwise = x : go (sucList g x ++ xs) (IS.insert x visited)

udfs :: Gr a -> [Node] -> [Node]
udfs g start = go start IS.empty
  where
    go [] visited = []
    go (x:xs) visited
      | IS.member x visited = go xs visited
      | otherwise = x : go (sucList g x ++ preList g x ++ xs) (IS.insert x visited)

topsort :: Gr a -> [Node]
topsort g = (foldr (.) id $ evalState (traverse go (nodes g)) IS.empty) []
  where
    go = \x -> do
      visited <- get
      case IS.member x visited of
        True -> pure id
        False -> do
          put (IS.insert x visited)
          before <- foldr (.) id <$> traverse go (preList g x)
          return (before . (x:))

subgraph :: Gr a -> [Node] -> Gr a
subgraph gr ns = (mkGraph
                  (filter (\(i,a) -> IS.member i nset) (labNodes gr))
                  (filter (\(i,j) -> IS.member i nset && IS.member j nset) (edges gr)))
  where
    nset = IS.fromList ns


-- Assumes that g is undirected, but does not check.
components :: Gr a -> [[Node]]
components g = filter (not . null) $ evalState (traverse go (nodes g)) IS.empty
  where
    go = \x -> go1 [x] []
    go1 :: [Node] -> [Node] -> State IntSet [Node]
    go1 [] os = pure os
    go1 (x:xs) os = do
      visited <- get
      case IS.member x visited of
        True -> go1 xs os
        False -> do
          put (IS.insert x visited)
          go1 (sucList g x ++ xs) (x:os)

splitComponents :: Gr a -> [Gr a]
splitComponents (Gr m) = [Gr (IM.restrictKeys m (IS.fromList c)) | c <- components (Gr m)]

isEmpty :: Gr a -> Bool
isEmpty (Gr m) = IM.null m

isConnected :: Gr a -> Bool
isConnected g = isEmpty g || IS.fromList (udfs g (take 1 (nodes g))) == IS.fromList (nodes g)

isUndirected :: Gr a -> Bool
isUndirected (Gr m) = all ok (toList m)
  where
    ok (p, a, s) = p == s

-- Make simple and undirected, remove labels
simplify :: Gr a -> Gr ()
simplify gr = const () <$> simplify' gr

-- Make simple and undirected
simplify' :: Gr a -> Gr a
simplify' gr = gmap dedup gr
  where
    dedup node (p, a, s) =
      let adj = IS.delete node $ (p <> s)
      in (adj, a, adj)

plot :: String -> Gr () -> IO ()
plot fname gr = writeFile fname txt
  where
    txt = unlines [
      "digraph {",
      unlines [show n | n <- nodes gr],
      unlines [show a ++ " -- " ++ show b | (a,b) <- edges gr, hasEdge gr (b,a), a < b],
      unlines [show a ++ " -> " ++ show b | (a,b) <- edges gr, not (hasEdge gr (b,a))],
      "}"]

plotLab :: Show a => String -> Gr a -> IO ()
plotLab fname gr = writeFile fname txt
  where
    txt = unlines [
      if isUndirected gr then "graph {" else "digraph {",
      unlines [printf "%i [label=\"%s\"]" n (show a) | (n,a) <- labNodes gr],
      unlines [show a ++ " -- " ++ show b | (a,b) <- edges gr, hasEdge gr (b,a), a < b],
      unlines [show a ++ " -> " ++ show b | (a,b) <- edges gr, not (hasEdge gr (b,a))],
      "}"]
