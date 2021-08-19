{-# Language BangPatterns #-}
module TreeWidth where

import           Data.Foldable
import           Data.IntMap (IntMap)
import qualified Data.IntMap as IM
import           Data.OrdPSQ (OrdPSQ)
import qualified Data.OrdPSQ as PQ
import           Data.IntSet (IntSet)
import qualified Data.IntSet as IS
import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.Set (Set)
import qualified Data.Set as Set
import           Data.Tuple
import           Debug.Trace

import           Graph (Gr, Node)
import qualified Graph as G

type Bag = IntSet

-- O(n^2 d^2), where d is the average degree
slowTreeWidth :: Gr a -> Gr Bag
slowTreeWidth ga = go (G.simplify ga) []
  where
    go gr ns = case min_fill_in gr of
                 Just node -> let gr' = G.insEdges [(a, b) |
                                                    a <- G.sucList gr node,
                                                    b <- G.sucList gr node,
                                                    a /= b,
                                                    not (G.hasEdge gr (a,b))] $ G.delNode node gr
                              in go gr' ((node, G.suc gr node) : ns)
                 Nothing -> finish ns (G.mkGraph [(0, IS.fromList (G.nodes gr))] [])
    finish [] tree = tree
    finish ((node,neighbors):ns) tree = finish ns tree'
      where
        target = head ([i | (i, bag) <- G.labNodes tree, IS.isSubsetOf neighbors bag] ++ [0]) -- inefficient
        [new_id] = G.newNodes 1 tree
        adj = IS.singleton target
        tree' = (adj, new_id, IS.insert node neighbors, adj) G.& tree

-- O(n d^2 log n), where d is the average degree
treeWidth :: Gr a -> Gr Bag
treeWidth ga = let gr = G.simplify ga in go gr [] (initCache gr)
  where
    go !gr !ns !cache =
      case minCache cache of
        Just node -> let neighbors = G.sucList gr node
                         newEdges = [(a, b) |
                                      a <- neighbors,
                                      b <- neighbors,
                                      a /= b,
                                      not (G.hasEdge gr (a,b))]
                         gr' = G.insEdges newEdges $ G.delNode node gr
                         dirty = IS.fromList $ [node] ++ neighbors ++ (neighbors >>= G.sucList gr)
                     in go gr' ((node, IS.fromList $ neighbors) : ns) (updateCache gr' dirty cache)
        Nothing -> finish ns (IS.fromList (G.nodes gr))
    finish ns initBag = gofinish ns (G.mkGraph [(0, initBag)] []) (IM.fromSet (const (IS.singleton 0)) initBag)
    -- 'bags' indexes the current bags of the tree by their contents
    -- bags : vertex v -> set of bags containing v
    gofinish [] tree bags = tree
    gofinish ((node,neighbors):ns) tree bags = gofinish ns tree' bags'
      where
        -- Either connect to some bag that contains all our neighbors, or connect to the first bag
        target = case [b | n <- IS.toList neighbors, Just b <- [IM.lookup n bags]] of
          [] -> 0
          bs -> head (IS.toList (foldr1 IS.intersection bs) ++ [0])
        [new_id] = G.newNodes 1 tree
        new_bag = IS.insert node neighbors
        adj = IS.singleton target
        tree' = (adj, new_id, new_bag, adj) G.& tree
        bags' = IM.unionWith (<>) bags (IM.fromSet (const (IS.singleton new_id)) new_bag)

-- The cache maintains a priority queue of nodes according to their
-- fill number, such that the node with minimum fill number can be
-- obtained in O(1), and the cache can be updated to accomodate
-- removal or addition of nodes in O(log n).
type MinFillCache = OrdPSQ Node (Int, Int, Node) ()

-- O(n d^2), where d is the average degree of nodes
initCache :: Gr a -> MinFillCache
initCache gr = PQ.fromList [(node, (fill gr node, G.outdeg gr node, node), ()) | node <- G.nodes gr]

-- O(m log n), where m is the number of nodes to refresh
updateCache :: Gr a -> IntSet -> MinFillCache -> MinFillCache
updateCache gr nodes cache = foldr update cache (IS.toList nodes)
  where
    update node cache
      | G.hasNode gr node = let f = fill gr node
                                d = G.outdeg gr node
                            in PQ.insert node (f, d, node) () cache
      | otherwise = PQ.delete node cache

-- O(log n)
minCache :: MinFillCache -> Maybe Node
minCache cache
  | order == 0 = Nothing
  | degree == order - 1 = Nothing
  | otherwise = Just n
  where
    order = PQ.size cache
    Just (n, (_, degree, _), _) = PQ.findMin cache

-- Find fill number of a node, defined as the minimum number of edges
-- that must be added to the graph to make the neighborhood of `node`
-- into a clique.  O(log n + e^2), where e is the number of neighbors
-- of node
fill :: Gr a -> Node -> Int
fill gr node = sum (map subfill (IS.toList neighbors)) `div` 2
  where neighbors = G.suc gr node
        subfill n = IS.size (neighbors `IS.difference` G.suc gr n) - 1

-- Find node with minimum fill number.
-- O(n d^2)
min_fill_in :: Gr a -> Maybe G.Node
min_fill_in gr
  | G.isEmpty gr = Nothing
  | degree == G.order gr - 1 = Nothing
  | otherwise = case minimum [(fill gr n, G.outdeg gr n, n) | n <- nodes] of
                  (_, _, node) -> Just node
  where
    nodes = G.nodes gr
    degree = minimum (map (G.outdeg gr) nodes)
