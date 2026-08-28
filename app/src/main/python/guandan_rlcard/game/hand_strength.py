"""Whole-hand strength and combination planning for Guandan agents.

The primary value is M-value: the fewest plays found to empty the hand.
For equal M-values we prefer decompositions that retain strong, coherent
combinations (bombs, straight flushes and compound combinations).  Search is
bounded so it remains usable during interactive play; small endgames are
normally solved exhaustively, while large hands use a beam.
"""

from dataclasses import dataclass
from functools import lru_cache

from guandan_rlcard.constants import CARD_RANK, CARD_RANK_INDEX
from guandan_rlcard.game.card_utils import card_from_str
from guandan_rlcard.game.hand_heuristics import estimate_min_steps
from guandan_rlcard.game.judger import GuandanJudger


DEFAULT_STRENGTH_WEIGHTS = {
    'four_bomb_circle': 0.78,
    'bomb_extra_card_circle': 0.18,
    'straight_flush_circle': 1.05,
    'joker_bomb_circle': 1.65,
    'joker_single_circle': 0.72,
    'joker_single_rescue_circle': 0.48,
    'high_card_circle': 0.10,
    'high_compound_circle': 0.07,
}
_strength_weights = dict(DEFAULT_STRENGTH_WEIGHTS)


def configure_strength_weights(weights=None):
    """Install trainable hand-strength weights and clear dependent caches."""
    global _strength_weights
    merged = dict(DEFAULT_STRENGTH_WEIGHTS)
    if weights:
        unknown = set(weights) - set(merged)
        if unknown:
            raise ValueError(f'Unknown hand-strength weights: {unknown}')
        merged.update({key: float(value) for key, value in weights.items()})
    _strength_weights = merged
    _evaluate_cached.cache_clear()
    return dict(_strength_weights)


def current_strength_weights():
    return dict(_strength_weights)


@dataclass(frozen=True)
class HandStrength:
    min_turns: int
    effective_turns: float
    initiative_circles: float
    power: float
    combination_value: float
    plan: tuple
    exact: bool


def _state(cards):
    return tuple(sorted(cards))


def _remove(state, cards):
    rest = list(state)
    for card in cards:
        try:
            rest.remove(card)
        except ValueError:
            return None
    return tuple(rest)


def _rank_value(rank, level_rank_index):
    if rank == 'R':
        return 17
    if rank == 'B':
        return 16
    if rank == CARD_RANK[level_rank_index]:
        return 15
    return CARD_RANK_INDEX.get(rank, 0) + 2


def _combo_value(action, level_rank_index):
    kind, rank, cards = action
    base = {
        'Single': 0, 'Pair': 3, 'Trips': 7, 'Straight': 14,
        'ThreeWithTwo': 17, 'ThreePair': 22, 'TwoTrips': 24,
        'StraightFlush': 48, 'Bomb': 55,
    }.get(kind, 0)
    if kind == 'Bomb':
        base += max(0, len(cards) - 4) * 14
    return base + _rank_value(rank, level_rank_index) * 0.12


def _initiative_value(action, level_rank_index):
    """Estimated circles of initiative/control supplied by a combination.

    A control combination is not merely another play: if it wins the trick,
    it gives the player another lead.  Values deliberately distinguish a
    vulnerable four-card bomb from a large bomb or straight flush.
    """
    kind, rank, cards = action
    if kind == 'Single' and rank in ('B', 'R'):
        return _strength_weights['joker_single_circle']
    if kind == 'StraightFlush':
        return _strength_weights['straight_flush_circle']
    if kind == 'Bomb':
        if rank == 'R':
            return _strength_weights['joker_bomb_circle']
        return (_strength_weights['four_bomb_circle']
                + max(0, len(cards) - 4)
                * _strength_weights['bomb_extra_card_circle'])
    rank_value = _rank_value(rank, level_rank_index)
    if kind in ('Single', 'Pair', 'Trips'):
        return (max(0.0, rank_value - 13)
                * _strength_weights['high_card_circle'])
    if kind in ('ThreePair', 'TwoTrips', 'ThreeWithTwo'):
        return (max(0.0, rank_value - 12)
                * _strength_weights['high_compound_circle'])
    return 0.0


def _plan_metrics(plan, level_rank_index):
    initiative = sum(_initiative_value(action, level_rank_index)
                     for action in plan)
    # A separately played joker can win a trick and then provide a fresh lead
    # for one otherwise awkward low single.  Four jokers as one bomb provide
    # only one such return.  This feature lets training decide when spreading
    # the controls is worth the extra nominal plays.
    joker_singles = sum(
        1 for action in plan
        if action[0] == 'Single' and action[1] in ('B', 'R'))
    weak_singles = sum(
        1 for action in plan
        if action[0] == 'Single'
        and action[1] not in ('B', 'R')
        and _rank_value(action[1], level_rank_index) <= 12)
    initiative += (min(joker_singles, weak_singles)
                   * _strength_weights['joker_single_rescue_circle'])
    # One control circle can justify roughly one extra nominal play.  This
    # prevents a shorter decomposition from automatically destroying a bomb
    # or straight flush, while still preferring genuinely compact hands.
    effective = len(plan) - initiative
    value = sum(_combo_value(action, level_rank_index) for action in plan)
    return effective, initiative, value


def _power(cards, level_rank_index):
    """Approximate control-card power of a remaining hand."""
    level = CARD_RANK[level_rank_index]
    score = 0.0
    for card in cards:
        rank = card[-1]
        if rank == 'R':
            score += 2.0
        elif rank == 'B':
            score += 1.5
        elif rank == level:
            score += 1.2 if card[0] == 'H' else 0.9
        elif rank in ('A', 'K'):
            score += 0.45 if rank == 'A' else 0.25
    return score


@lru_cache(maxsize=12000)
def _lead_actions(state, level_rank_index):
    objects = [card_from_str(card) for card in state]
    generated = GuandanJudger.playable_actions_from_hand(
        objects, level_rank_index)
    # Different suit choices can produce the same remaining hand.  Keep one.
    unique = {}
    for action in generated:
        rest = _remove(state, action[2])
        if rest is None or rest == state:
            continue
        key = (action[0], action[1], rest)
        candidate = (action[0], action[1], tuple(action[2]))
        old = unique.get(key)
        if old is None or candidate[2] < old[2]:
            unique[key] = candidate
    return tuple(unique.values())


def _search_plan(cards, level_rank_index, beam_width=64, action_limit=50):
    start = _state(cards)
    if not start:
        return (), True

    strategic_alternatives = []
    four_jokers_at_start = start.count('SB') == 2 and start.count('HR') == 2
    if four_jokers_at_start:
        non_jokers = list(start)
        for card in ('SB', 'SB', 'HR', 'HR'):
            non_jokers.remove(card)
        rest_plan, rest_exact = _search_plan(
            non_jokers, level_rank_index, beam_width, action_limit)
        split_controls = (
            ('Single', 'B', ('SB',)),
            ('Single', 'B', ('SB',)),
            ('Single', 'R', ('HR',)),
            ('Single', 'R', ('HR',)),
        )
        strategic_alternatives.append((split_controls + rest_plan,
                                       rest_exact))

    # (state, plan, accumulated decomposition value)
    frontier = [(start, (), 0.0)]
    seen_depth = {start: 0}
    max_depth = max(1, min(len(start),
                           estimate_min_steps(list(start), level_rank_index) + 5))
    truncated = False
    solutions = []
    first_solution_depth = None

    for depth in range(1, max_depth + 1):
        next_states = {}
        for state, plan, value in frontier:
            actions = list(_lead_actions(state, level_rank_index))
            actions.sort(
                key=lambda a: (len(a[2]), _combo_value(a, level_rank_index)),
                reverse=True)
            if len(actions) > action_limit:
                actions = actions[:action_limit]
                truncated = True
            for action in actions:
                rest = _remove(state, action[2])
                if rest is None:
                    continue
                new_plan = plan + (action,)
                new_value = value + _combo_value(action, level_rank_index)
                if not rest:
                    solutions.append(new_plan)
                    if first_solution_depth is None:
                        first_solution_depth = depth
                    continue
                if seen_depth.get(rest, 10 ** 6) < depth:
                    continue
                seen_depth[rest] = depth
                previous = next_states.get(rest)
                if previous is None or new_value > previous[2]:
                    next_states[rest] = (rest, new_plan, new_value)

        candidates = list(next_states.values())
        candidates.sort(
            key=lambda node: (
                estimate_min_steps(list(node[0]), level_rank_index),
                len(node[0]),
                -node[2],
                -_power(node[0], level_rank_index),
            ))
        if len(candidates) > beam_width:
            candidates = candidates[:beam_width]
            truncated = True
        frontier = candidates
        # Compare the minimum-turn solution with plans one play longer.  A
        # longer plan may be strategically superior if it preserves a bomb or
        # straight flush and therefore an additional circle of initiative.
        comparison_depth = 3 if four_jokers_at_start else 1
        if (first_solution_depth is not None
                and depth >= first_solution_depth + comparison_depth):
            break
        if not frontier:
            break


    solutions.extend(plan for plan, _ in strategic_alternatives)
    if solutions:
        best = min(
            solutions,
            key=lambda plan: (
                _plan_metrics(plan, level_rank_index)[0],
                len(plan),
                -_plan_metrics(plan, level_rank_index)[2],
            ),
        )
        alternative_exact = all(exact for _, exact in strategic_alternatives)
        return best, not truncated and alternative_exact

    # Safe fallback: singles always form a legal complete plan.
    fallback = tuple(('Single', card[-1], (card,)) for card in start)
    return fallback, False


@lru_cache(maxsize=4096)
def _evaluate_cached(state, level_rank_index):
    plan, exact = _search_plan(state, level_rank_index)
    effective, initiative, value = _plan_metrics(plan, level_rank_index)
    return HandStrength(
        min_turns=len(plan),
        effective_turns=effective,
        initiative_circles=initiative,
        power=_power(state, level_rank_index),
        combination_value=value,
        plan=plan,
        exact=exact,
    )


def evaluate_hand(cards, level_rank_index):
    """Recalculate M-value, control power and the best found decomposition."""
    return _evaluate_cached(_state(cards), level_rank_index)


def best_lead_from_plan(cards, legal_actions, level_rank_index):
    """Choose a legal lead belonging to the best whole-hand decomposition.

    Lead the least valuable member of the plan first, preserving control cards
    for later.  If the complete hand is one legal action, finish immediately.
    """
    strength = evaluate_hand(cards, level_rank_index)
    legal = [a for a in legal_actions if a[0] != 'PASS']
    if not legal or not strength.plan:
        return None
    if len(strength.plan) == 1:
        target = strength.plan[0]
    else:
        target = min(strength.plan,
                     key=lambda a: _combo_value(a, level_rank_index))
    target_cards = sorted(target[2])
    matches = [a for a in legal
               if a[0] == target[0] and sorted(a[2]) == target_cards]
    return matches[0] if matches else None
