module TestTreeWidth where

import           Data.Foldable
import           Data.IntMap (IntMap)
import qualified Data.IntMap as IM
import           Data.IntSet (IntSet)
import qualified Data.IntSet as IS
import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.Set (Set)
import qualified Data.Set as Set
import           Data.Tuple
import           Debug.Trace

import           Test.QuickCheck
import           Test.Tasty
import           Test.Tasty.QuickCheck

import           Graph (Gr, Node)
import qualified Graph as G
import           TestGraph
import           TreeWidth

-- Verify the three properties of a tree decomposition:
-- 1. The union of all bags = the set of nodes
check1 :: Gr () -> Bool
check1 gr = IS.unions (map snd (G.labNodes (treeWidth gr))) == IS.fromList (G.nodes gr)

-- 2. For every edge (a,b), there is a bag which includes both vertices.
check2 :: Gr () -> Bool
check2 gr = let tree = treeWidth gr
                sets = map snd (G.labNodes tree)
            in and [any (\s -> IS.member a s && IS.member b s) sets | (a, b) <- G.edges gr]

-- 3. For a given vertex v, bags containing v are connected.
check3 :: Gr () -> Bool
check3 gr = let tree = treeWidth gr
            in and [G.isConnected (G.labfilter (IS.member v) tree) | v <- G.nodes gr]

-- ?. should validate that the result is a tree?
check4 :: Gr () -> Bool
check4 gr = let tree = treeWidth gr
            in length (G.edges tree) == 2 * (G.order tree - 1)

testTreeWidth :: TestTree
testTreeWidth = testGroup "TreeWidth" [
  testProperty "vertices" check1,
  testProperty "edges" check2,
  testProperty "connected" check3,
  testProperty "tree" check4
  ]
