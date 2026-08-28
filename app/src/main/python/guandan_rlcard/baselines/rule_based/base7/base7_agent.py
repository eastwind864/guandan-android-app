# -*- coding: utf-8 -*-
# @Time       : 2020/10/19 19:30
# @Author     : Duofeng Wu  &&  Zenghui Qian


from guandan_rlcard.game.player import GuandanPlayer as Player
from guandan_rlcard.game.hand_strength import (
    best_lead_from_plan,
)
from guandan_rlcard.game.hand_heuristics import estimate_min_steps
from collections import Counter
import json

# 中英文对照表
ENG2CH = {
    "Single": "单张",
    "Pair": "对子",
    "Trips": "三张",
    "ThreePair": "三连对",
    "ThreeWithTwo": "三带二",
    "TwoTrips": "钢板",
    "Straight": "顺子",
    "StraightFlush": "同花顺",
    "Bomb": "炸弹",
    "PASS": "过"
}

CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
            'A', 'BJ', 'RJ']

#              0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
str_to_ind = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'L', 'B', 'R']
#              0    1    2    3
str_to_flo = ['S', 'H', 'C', 'D']


class Action(object):

    def __init__(self):
        self.action = []
        self.act_range = -1

    def parse(self, msg, mate_pos):     # 增加了一个新参数mate_pos，表示队友的位置
        
        act = solve(msg, mate_pos) # 注意，此处返回的直接是动作而非序号了
        return act


class Base7Agent(Player):
    ''' Baseline 7 agent.
    '''
    name = 'Base7'
    
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.action = Action()
        self.my_pos = self.player_id                   # 增加了一个属性，用来记录自己的位置
        self.mate_pos = (self.my_pos + 2) % 4          # 增加了一个属性，用来记录队友的位置
        self.use_raw = True

    def step(self, state):
        if not state['actions']:
            return []
        
        msg = self.parse(state)
        act = self.action.parse(msg, self.mate_pos)

        # The historical formula mostly values the immediate response.  Add
        # a whole-hand plan so a 44+555 full house is preferred on lead and
        # KKK/AAAA is not split merely to cover an ordinary single.
        act = improve_action_for_hand_plan(act, msg, self.mate_pos)

        # The historical Base7 formula rewards bombs by their card count but
        # does not charge for heart-level wildcards.  It can therefore spend
        # both valuable wildcards on an 8-card bomb merely to cover an
        # ordinary combo.  Preserve that bomb unless it is an endgame push or
        # an opponent is close to going out.
        if should_conserve_wildcard_bomb(act, msg, self.mate_pos):
            pass_action = next(
                (candidate for candidate in state['actions']
                 if candidate[0] == 'PASS'),
                None,
            )
            if pass_action is not None:
                return pass_action
             
        return act
    
    def parse(self, state):
        assert type(state) == dict
        msg = {}
        rank_list = state['rank_list']
        play_team = state['play_team']
        msg['curPos'] = self.player_id
        msg['indexRange'] = len(state["actions"]) - 1
        msg['curAction'] = state['greaterAction']
        msg['curRank'] = CARD_RANK[rank_list[play_team]]
        msg['remain_cards'] = state['remain_cards']
        msg['pass_num'] = state['pass_num'][self.player_id]
        msg['my_pass_num'] = state['my_pass_num'][self.player_id]
        msg['history'] =  {
            '0': {
                'send': state['played_cards'][0],
                'remain': state['num_cards_left'][0],
            },
            '1': {
                'send': state['played_cards'][1],
                'remain': state['num_cards_left'][1],
            },
            '2': {
                'send': state['played_cards'][2],
                'remain': state['num_cards_left'][2],
            },
            '3': {
                'send': state['played_cards'][3],
                'remain': state['num_cards_left'][3],
            },
        }
        msg["greaterPos"] = state['greaterPos']
        msg["greaterAction"] = state['greaterAction']
        msg["actionList"] = state['actions']
        msg["handCards"] = state['current_hand']
        msg["publicInfo"] = {0:{}, 1:{}, 2:{}, 3:{}}
        for i in range(4):
            msg['publicInfo'][i]['rest'] = state['num_cards_left'][i]
            
        return msg

    def tribute_act(self, actionLists, rank):
        """Choose and execute a legal tribute card."""
        act = _choose_exchange_action(self, actionLists, rank)
        self.execute_tribute(act)
        return act

    def back_act(self, actionLists, rank, tribute_result):
        """Choose and execute a legal return-tribute card."""
        act = _choose_exchange_action(self, actionLists, rank)
        self.execute_back(act)
        return act


def should_conserve_wildcard_bomb(action, msg, mate_pos):
    """Avoid wasting heart-level wildcards on a non-urgent ordinary play."""
    if not action or action[0] != 'Bomb' or not action[2]:
        return False

    greater_action = msg.get('greaterAction') or []
    if not greater_action or greater_action[0] in ('Bomb', 'StraightFlush'):
        return False

    greater_pos = msg.get('greaterPos', -1)
    if greater_pos in (-1, mate_pos):
        return False

    wildcard = 'H' + msg.get('curRank', '')
    if wildcard not in action[2]:
        return False

    hand_size = len(msg.get('handCards') or [])
    remaining_after = hand_size - len(action[2])
    opponent_left = (msg.get('publicInfo', {}).get(greater_pos, {})
                     .get('rest', 27))

    # Going out (or leaving a one/two-card finish) and stopping an opponent
    # with five or fewer cards are urgent enough to justify the wildcard bomb.
    return remaining_after > 2 and opponent_left > 5


def _choose_exchange_action(player, action_lists, rank):
    """Use Base7's card-value model for tribute and return tribute."""
    if not action_lists:
        return []
    has = [
        [0] * 14,
        [0] * 14,
        [0] * 13,
        [0] * 13,
        [0] * 13,
        [0] * 2,
    ]
    for card in player.current_hand_str:
        if card[1] == rank:
            has[5][1 if card[0] == 'H' else 0] += 1
        card_to_list(card, has, 1)

    def score(action):
        values = [getval(card, rank, has) for card in action[2]]
        return cac(1, max(values), 1) if values else -10 ** 6

    return max(action_lists, key=score)


def _remaining_after_action(hand_cards, action):
    remaining = list(hand_cards)
    if not action or action[0] == 'PASS':
        return remaining
    for card in action[2]:
        try:
            remaining.remove(card)
        except ValueError:
            return None
    return remaining


def _projected_turns(hand_cards, action, level_rank_index):
    """Estimate total turns from now, including the proposed play."""
    if action[0] == 'PASS':
        return estimate_min_steps(hand_cards, level_rank_index)
    remaining = _remaining_after_action(hand_cards, action)
    if remaining is None:
        return 10 ** 6
    return 1 + estimate_min_steps(remaining, level_rank_index)


def _split_penalty(hand_cards, action):
    """Charge for taking only part of an existing same-rank group."""
    if not action or action[0] == 'PASS':
        return 0
    hand_counts = Counter(card[-1] for card in hand_cards)
    used_counts = Counter(card[-1] for card in action[2])
    return sum(1 for rank, used in used_counts.items()
               if hand_counts[rank] >= 2 and used < hand_counts[rank])


def _plan_cost(hand_cards, action, level_rank_index):
    return (_projected_turns(hand_cards, action, level_rank_index)
            + _split_penalty(hand_cards, action))


def improve_action_for_hand_plan(preferred, msg, mate_pos):
    """Protect useful combinations by minimising plays-to-empty.

    Base7's original point formula can optimise the current trick while
    turning one compact combination into several later plays.  This planner
    keeps the original choice as a tie-breaker, but replaces it when another
    legal play gives a strictly shorter route to an empty hand.  When
    following an ordinary opponent play it may pass instead of making the
    route longer.  Endgame defence remains with the original tactical logic.
    """
    actions = msg.get('actionList') or []
    hand = msg.get('handCards') or []
    if not preferred or not actions or not hand:
        return preferred

    try:
        level_rank_index = CARD_RANK.index(msg.get('curRank'))
    except ValueError:
        return preferred

    plays = [action for action in actions if action[0] != 'PASS']
    if not plays:
        return preferred

    # On a fresh trick, use the decomposition itself.  It is recomputed from
    # the current hand after every play, so the plan adapts automatically.
    greater_action = msg.get('greaterAction') or []
    if not greater_action:
        planned = best_lead_from_plan(hand, plays, level_rank_index)
        if planned is not None:
            return planned

    projections = {
        id(action): _plan_cost(hand, action, level_rank_index)
        for action in plays
    }
    best_turns = min(projections.values())
    best_plays = [action for action in plays
                  if projections[id(action)] == best_turns]

    # Preserve the historical tactical choice whenever it is equally good.
    if (preferred[0] != 'PASS'
            and _plan_cost(
                hand, preferred, level_rank_index) == best_turns):
        best = preferred
    else:
        best = max(best_plays, key=lambda action: len(action[2]))

    if not greater_action:
        return best

    greater_pos = msg.get('greaterPos', -1)
    if greater_pos == mate_pos:
        return preferred
    opponent_left = (msg.get('publicInfo', {}).get(greater_pos, {})
                     .get('rest', 27))
    if opponent_left <= 2:
        return preferred
    pass_action = next(
        (action for action in actions if action[0] == 'PASS'), None)

    # Following play needs tactical restraint.  Do not globally optimise the
    # approximate step estimate (that makes the bot pass too often); protect
    # only the clear case of breaking a triple/bomb for one ordinary single.
    if (pass_action is not None
            and greater_action[0] == 'Single'
            and preferred[0] == 'Single'):
        rank = preferred[2][0][-1]
        rank_count = sum(card[-1] == rank for card in hand)
        if rank_count >= 3:
            return pass_action
    return preferred


    def tribute_act(self, actionLists, rank):
        """根据合法动作集选择动作执行"""
        zero_s_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # 2 -- 王
        zero_h_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_c_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # 2 -- A
        zero_d_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_number_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # 2 -- A
        zero_rank_cards = [0, 0]            # 非红桃 红桃

        has = [zero_s_cards, zero_h_cards, zero_c_cards, zero_d_cards, zero_number_cards, zero_rank_cards]
        now_rank = rank           # 获取等级

        for card in self.current_hand_str:       # 统计当前手牌
            if card[1] == now_rank and card[0] == 'H':
                has[5][1] += 1
            elif card[1] == now_rank and card[0] != 'H':
                has[5][0] += 1
            card_to_list(card, has, 1)

        act_score = []                  # 存放所有行动选项的评分
        for action in actionLists:
            values = []
            for one in action[2]:
                values.append(getval(one, now_rank, has))
            value = max(values)
            poss = 1
            score = cac(1, value, poss)
            act_score.append(score)
            
        act_index = act_score.index(max(act_score))
        act = actionLists[act_index]
        self.execute_tribute(act)
        
        return act
    
    def back_act(self, actionLists, rank, tribute_result):
        """根据合法动作集选择动作执行"""
        zero_s_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # 2 -- 王
        zero_h_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_c_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # 2 -- A
        zero_d_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_number_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # 2 -- A
        zero_rank_cards = [0, 0]            # 非红桃 红桃

        has = [zero_s_cards, zero_h_cards, zero_c_cards, zero_d_cards, zero_number_cards, zero_rank_cards]
        now_rank = rank           # 获取等级

        for card in self.current_hand_str:       # 统计当前手牌
            if card[1] == now_rank and card[0] == 'H':
                has[5][1] += 1
            elif card[1] == now_rank and card[0] != 'H':
                has[5][0] += 1
            card_to_list(card, has, 1)

        act_score = []                  # 存放所有行动选项的评分
        for action in actionLists:
            values = []
            for one in action[2]:
                values.append(getval(one, now_rank, has))
            value = max(values)
            poss = 1
            score = cac(1, value, poss)
            act_score.append(score)
        
        act_index = act_score.index(max(act_score))
        act = actionLists[act_index]
        self.execute_back(act)
        
        return act

def card_to_list(card, my_list, step):
    des = str_to_flo.index(card[0])
    #  数字
    if card[1] == 'T':
        my_list[des][8] += step
        my_list[4][8] += step
    elif card[1] == 'J':
        my_list[des][9] += step
        my_list[4][9] += step
    elif card[1] == 'Q':
        my_list[des][10] += step
        my_list[4][10] += step
    elif card[1] == 'K':
        my_list[des][11] += step
        my_list[4][11] += step
    elif card[1] == 'A':
        my_list[des][12] += step
        my_list[4][12] += step
    elif card[1] == 'B':
        my_list[des][13] += step
    elif card[1] == 'R':
        my_list[des][13] += step
    else:
        my_list[des][int(card[1]) - 2] += step
        my_list[4][int(card[1]) - 2] += step


def getval(card, rank, has):               # 评估一张牌在手牌中的价值，card: 牌面 如"H2"，rank 当前等级，has 手牌情况
    des = str_to_flo.index(card[0])
    val = str_to_ind.index(card[1]) + 1
    index = str_to_ind.index(card[1])
    if card[1] == 'B' or card[1] == 'R':
        index = 13
    if card[1] == rank:                     # 对级牌重新赋予价值
        val = 14
    if card == 'SB':                        # 对大王小王重新赋予价值
        if has[0][13] == 2 and has[1][13] == 2:
            val *= 100
        elif has[0][13] == 2 and has[1][13] != 2:
            val += 20
    elif card == 'HR':
        if has[0][13] == 2 and has[1][13] == 2:
            val *= 100
        elif has[0][13] != 2 and has[1][13] == 2:
            val += 20
    elif card[0] == 'H' and card[1] == rank:    # 对红桃级牌重新赋予价值
        val = 340
    else:
        ans = 0                         # ans 暂时记录构成同花顺的牌的新价值
        for num in range(index-4, index+1):   # 搜寻同花顺
            if num >= -1 and num+4 <= 12:
                if num == -1:
                    if has[des][12] >= 1 and has[des][0] >= 1 and has[des][1] >= 1 and has[des][2] >= 1 \
                            and has[des][3] >= 1:
                        ans = 320+val
                else:
                    if has[des][num] >= 1 and has[des][num+1] >= 1 and has[des][num+2] >= 1 and has[des][num+3] >= 1 \
                            and has[des][num+4] >= 1:
                        ans = 320+val
        if has[4][index] <= 3:          # 查看同点数的牌的数目，根据数目重新赋予价值
            val += 20 * (has[4][index]-1)
        elif has[4][index] == 4:
            val += 220
        elif has[4][index] == 5:
            val += 300
        elif has[4][index] == 6:        # 修改了超过5张的赋值，是更符合出牌逻辑 2020.10.21
            val += 400                  # 再次修改  2020.10.23
        elif has[4][index] == 7:
            val += 500
        elif has[4][index] == 8:
            val += 600
        if val < ans:                   # 取val 和 ans 中较大者为最终价值
            val = ans
    return val                          # 评估完成，返回该牌的价值


def cac(gain, value, poss):             # 评分计算公式 有待完善
    return gain*(1 + poss)/value        # Score = Gain * ( 1 + Possibility ) / Value;（算法会优先打出评分较高的打法）


def solve(msg, mate_pos):                   # 主体函数  生成记录手牌情况的列表
    zero_s_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # 2 -- 王
    zero_h_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    zero_c_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # 2 -- A
    zero_d_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    zero_number_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # 2 -- A
    zero_rank_cards = [0, 0]            # 非红桃 红桃

    has = [zero_s_cards, zero_h_cards, zero_c_cards, zero_d_cards, zero_number_cards, zero_rank_cards]
    now_rank = msg["curRank"]           # 获取等级
    now_greater_pos = msg["greaterPos"]
    op1_pos = (mate_pos + 1) % 4
    op2_pos = (mate_pos + 3) % 4

    for card in msg["handCards"]:       # 统计当前手牌
        if card[1] == now_rank and card[0] == 'H':
            has[5][1] += 1
        elif card[1] == now_rank and card[0] != 'H':
            has[5][0] += 1
        card_to_list(card, has, 1)

    act_score = []                  # 存放所有行动选项的评分

    for action in msg["actionList"]:   # 对所有行动选项进行评估
        if len(action[2]) == len(msg["handCards"]):
            act_score.append(100000)
        else:
            if action[0] == "Single":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(1, value, poss)
                n_score = score
                if now_greater_pos == -1 or now_greater_pos == msg["curPos"]:
                    if msg["publicInfo"][op1_pos]['rest'] == 1:
                        n_score = -1 * score
                    if msg["publicInfo"][mate_pos]['rest'] == 1:
                        n_score = score + 100
                    if msg["publicInfo"][op2_pos]['rest'] == 1:
                        n_score = -1 * score
                    act_score.append(n_score)
                else:
                    if msg["publicInfo"][op1_pos]['rest'] == 1:
                        n_score = score
                    if msg["publicInfo"][mate_pos]['rest'] == 1:
                        n_score = -10000
                    if msg["publicInfo"][op2_pos]['rest'] == 1:
                        n_score = -1 * score
                    act_score.append(n_score)

            elif action[0] == "Pair":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(2, value, poss)
                n_score = score
                if now_greater_pos == -1 or now_greater_pos == msg["curPos"]:
                    if msg["publicInfo"][op1_pos]['rest'] == 2:
                        n_score = -1 * score
                    if msg["publicInfo"][mate_pos]['rest'] == 2:
                        n_score = score + 100
                    if msg["publicInfo"][op2_pos]['rest'] == 2:
                        n_score = -1 * score
                    act_score.append(n_score)
                else:
                    if msg["publicInfo"][op2_pos]['rest'] == 2 or msg["publicInfo"][op2_pos]['rest'] == 2:
                        n_score = -1 * score
                    act_score.append(n_score)

            elif action[0] == "Trips":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(3, value, poss)
                n_score = score
                if now_greater_pos == -1 or now_greater_pos == msg["curPos"]:
                    if msg["publicInfo"][op1_pos]['rest'] == 3:
                        n_score = -1 * score
                    if msg["publicInfo"][op2_pos]['rest'] == 3:
                        n_score = -1 * score
                    act_score.append(n_score)
                else:
                    if msg["publicInfo"][op1_pos]['rest'] == 3 or msg["publicInfo"][op2_pos]['rest'] == 3:
                        n_score = -1 * score
                    act_score.append(n_score)

            elif action[0] == "ThreePair":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(6, value, poss)
                act_score.append(score)

            elif action[0] == "ThreeWithTwo":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = (max(values) + min(values)) / 2
                poss = 1
                score = cac(5, value, poss)
                act_score.append(score)

            elif action[0] == "TwoTrips":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(6, value, poss)
                act_score.append(score)

            elif action[0] == "Straight":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = sum(values)
                poss = 1
                score = cac(5, value, poss)
                act_score.append(score)

            elif action[0] == "StraightFlush":
                if now_greater_pos == mate_pos or now_greater_pos == -1 or now_greater_pos == msg["curPos"]:
                    act_score.append(-10000)    # 如果是队友，就不会出同花顺  解决 接队友单张时不能排除的问题  2020.10.20
                elif now_greater_pos == op1_pos or now_greater_pos == op2_pos:
                    if msg["publicInfo"][now_greater_pos]['rest'] <= 14:
                        values = []
                        for one in action[2]:
                            values.append(getval(one, now_rank, has))
                        value = max(values)
                        poss = 1
                        score = cac(5, value, poss)
                        act_score.append(score)
                    else:
                        act_score.append(-1)

            elif action[0] == "Bomb":
                if now_greater_pos == mate_pos or now_greater_pos == -1 or now_greater_pos == msg["curPos"]:
                    act_score.append(-10000)    # 如果是队友，就不会出炸弹    解决 接队友单张时不能排除的问题  2020.10.20
                elif now_greater_pos == op1_pos or now_greater_pos == op2_pos:
                    if msg["publicInfo"][now_greater_pos]['rest'] <= 14:
                        values = []
                        for one in action[2]:
                            values.append(getval(one, now_rank, has))
                        value = max(values)
                        poss = 1
                        score = cac(len(values), value, poss)
                        act_score.append(score)
                    else:
                        act_score.append(-1)

            elif action[0] == "PASS":
                if now_greater_pos == mate_pos:
                    if msg["publicInfo"][mate_pos]['rest'] <= 6:  # 队友的牌数小于6时，不会压队友  PASS优先级最高
                        value = 1
                        poss = 1
                        score = cac(1, value, poss)
                        act_score.append(score)
                    else:
                        if msg["greaterAction"][0] == "Single":  # 单张情况  酌情设置PASS优先级
                            value = 25
                            poss = 1
                            score = cac(2, value, poss)
                            act_score.append(score)
                        elif msg["greaterAction"][0] == "Pair":  # 对子情况  酌情设置PASS优先级
                            value = 65
                            poss = 1
                            score = cac(4, value, poss)
                            act_score.append(score)
                        elif msg["greaterAction"][0] == "Trips":  # 对子情况  酌情设置PASS优先级
                            value = 103
                            poss = 1
                            score = cac(6, value, poss)
                            act_score.append(score)
                        else:                               # 不会压队友  PASS优先级最高
                            value = 1
                            poss = 1
                            score = cac(1, value, poss)
                            act_score.append(score)
                else:                                       # 对方牌权  PASS优先级最低
                    if 1 <= msg["publicInfo"][op2_pos]['rest'] <= 3 :
                        act_score.append(-9999)
                    else:
                        value = 1
                        poss = 1
                        score = cac(0, value, poss)
                        act_score.append(score)

            elif action[0] == "back":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(1, value, poss)
                act_score.append(score)

            elif action[0] == "tribute":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(1, value, poss)
                act_score.append(score)

    result_index = act_score.index(max(act_score))     # 返回所有选项中分数最高的选项的下标

    return msg["actionList"][result_index]


