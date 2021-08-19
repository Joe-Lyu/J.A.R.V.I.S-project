module Util where

import           Data.Foldable
import           Data.List
import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.Ord
import           Data.Set (Set)
import qualified Data.Set as Set
import           Data.Tuple
import           Debug.Trace

reflex :: Ord a => [a] -> Map a a
reflex xs = Map.fromList [(x, x) | x <- xs]

memo :: Ord a => [a] -> (a -> b) -> (a -> b)
memo xs f = \x -> tab Map.! x
  where
    tab = f <$> reflex xs

minimumOn :: (Foldable t, Ord a) => (b -> a) -> t b -> b
minimumOn f xs = minimumBy (comparing f) xs

maximumOn :: (Foldable t, Ord a) => (b -> a) -> t b -> b
maximumOn f xs = maximumBy (comparing f) xs
