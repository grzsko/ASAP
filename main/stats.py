import json
from typing import Any, Optional, Tuple, TypeVar

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import spearmanr
from sklearn import metrics

from main.data import Mol
from shared import Db, Fn

from .ai import ai_input_gen
from .score import Score
from .tree import JsonTree, Tree, TreeTypes, sum_tree_stats
from .types import AiInput
from .utils import flatten, fn_txt, not_none, serialize_dict

matplotlib.rc("font", size=5)

T = TypeVar("T")


def avg(l: list[float]):
    return sum(l) / len(l)


AggFn = Fn[list[float], float]
AggTuple = Tuple[str, str, AggFn]

# name, color, scatter_kwargs, fn
agg_list: list[AggTuple] = [
    ("min", "blue", min),
    ("max", "red", max),
    ("avg", "green", avg),
]

ScoreGetter = Optional[Fn[Score, float]]
_score_getters: list[Tuple[str, ScoreGetter]] = [*Score.getters(), ("ai", None)]


def _unzip(l: list[Tuple[T, T]]):
    return ([x for x, _ in l], [y for _, y in l])


def _all_nodes(roots: list[Tree]):
    return flatten((root.all_nodes() for root in roots))


def _pairs_siblings(roots: list[Tree], xtype: list[TreeTypes], ytype: list[TreeTypes]):
    def pair(node: Tree):
        return (
            (x, y)
            for x in node.children
            if x.type in xtype
            for y in node.children
            if y.type in ytype
        )

    return list(flatten((pair(node) for node in _all_nodes(roots))))


def _pairs_parent_child(
    roots: list[Tree], parenttype: list[TreeTypes], childtype: list[TreeTypes]
):
    def pair(p: Tree):
        return ((p, c) for c in p.children if c.type in childtype)

    return list(
        flatten(
            (pair(parent) for parent in _all_nodes(roots) if parent.type in parenttype)
        )
    )


def display_str(data: str):
    display(HTML(f"<pre>{data}</pre>"))


def _ax6(title: str, has_data: bool):
    if has_data:
        fig, axs = plt.subplots(nrows=3, ncols=2)
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.75)
        fig.set_dpi(220)
        fig.set_figheight(7)
        fig.suptitle(title, horizontalalignment="left", x=0.005)
        for (score_name, getter), ax in zip(_score_getters, axs.flat):
            ax.set_title(f"score={score_name}")
            ax.ticklabel_format(style="plain")
            yield (score_name, getter, ax)
        plt.show()
    else:
        display_str(f"too little data for: \n{title}")


def _ax2(title: str, ttype: list[TreeTypes], btype: list[TreeTypes]):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.75)
    fig.set_dpi(220)
    fig.set_figheight(7)
    fig.suptitle(title, horizontalalignment="left", x=0.005)
    for type, ax in zip([ttype, btype], axs.flat):
        ax.set_title(f"type={type}")
        ax.ticklabel_format(style="plain")
        yield (type, ax)
    plt.show()


def _hist(
    ax: Any,
    x: list[float],
    bin_count: int,
    color: str,
    range: Optional[Tuple[float, float]] = None,
):
    return ax.hist(x=[x], color=[color], bins=bin_count, range=range)
    # ax.set_xticks(bins)
    """
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        percent = "%0.0f%%" % (100 * float(count) / counts.sum())
        ax.annotate(
            percent,
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -18),
            textcoords="offset points",
            va="top",
            ha="center",
        )
    """


def input_data(detailed: bool):
    def f(db: Db, ai_input: AiInput, mol: Mol):
        json_tree = db.read(["ai_postprocess", ai_input], JsonTree)
        if json_tree:
            return Tree(json_tree), mol

    data = list(
        not_none(f(db, ai_input, mol) for db, ai_input, mol in ai_input_gen(True, True))
    )
    solved_count = sum((root.type == "internal" for root, _mol in data))
    not_solved_count = sum((root.type == "not_solved" for root, _mol in data))
    all_count = len(data)
    assert sum((root.type == "solved" for root, _mol in data)) == 0
    assert (solved_count + not_solved_count) == all_count
    sum_stats = sum_tree_stats([root.stats() for root, _mol in data])
    yield (
        [root for root, _mol in data],
        f"ALL ({solved_count}/{all_count} solved); {json.dumps(sum_stats, indent=2)}",
    )
    if detailed:
        for root, mol in data:
            yield ([root], f"{mol.name}; {json.dumps(root.stats(), indent=2)}")


def _tree_to_float(
    tree: Tree,
    tree_to_scores: Fn[Tree, list[Score]],
    getter: ScoreGetter,
    agg_fn: AggFn,
) -> float:
    if getter:
        return agg_fn([getter(score) for score in tree_to_scores(tree)])
    else:
        return tree.ai_score


def scatter_pairs(
    xtype: list[TreeTypes],
    ytype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        xs, ys = _unzip(_pairs_siblings(roots, xtype, ytype))
        for agg_name, agg_color, agg_fn in agg_list:
            title = f"""
# Scatter plot of siblings nodes of types {xtype} and {ytype}.
Node score computed as {agg_name} over {fn_txt(tree_to_scores)} molecules.
"""
            for score_name, getter, ax in _ax6(title, bool(xs)):

                def stat(t: Tree):
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]
                rho, pval = spearmanr(x, y)
                info = {
                    "score": score_name,
                    "rho": round(rho, 10),
                    "pval": round(pval, 10),
                }
                _min = min([min(x), min(y)])
                _max = max([max(x), max(y)])
                ax.axis("square")
                ax.set_xlim([_min, _max])
                ax.set_ylim([_min, _max])
                ax.set_title(serialize_dict(info, ", "))
                ax.axline((_min, _min), (_max, _max), linewidth=1, color="black")
                ax.scatter(x, y, s=1, linewidths=1, color=agg_color)


def histogram_pairs_siblings(
    xtype: list[TreeTypes],
    ytype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        pairs = _pairs_siblings(roots, xtype, ytype)
        for agg_name, agg_color, agg_fn in agg_list:
            title = f"""
# Histogram of score differences between sibling nodes of types {xtype} and {ytype}.
Node computed as a {agg_name} over {fn_txt(tree_to_scores)}.
"""
            for _score_name, getter, ax in _ax6(title, bool(pairs)):

                def stat(t: Tree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                ax.xaxis.set_tick_params(labelsize="x-small")
                # ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
                _hist(
                    ax,
                    [stat(x) - stat(y) for x, y in pairs],
                    20,
                    agg_color,
                )
                # np.linspace(from, to, step)


def histogram_pairs_parent_child(
    parenttype: list[TreeTypes],
    childtype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        pairs = _pairs_parent_child(roots, parenttype, childtype)
        for agg_name, agg_color, agg_fn in agg_list:
            title = f"""
## Histogram of score differences between parent node of type {parenttype} and children node of type {ytype}.
Node computed as a {agg_name} over {fn_txt(tree_to_scores)}.
"""
            for _score_name, getter, ax in _ax6(title, bool(pairs)):

                def stat(t: Tree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                ax.xaxis.set_tick_params(labelsize="x-small")
                # ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
                _hist(
                    ax,
                    [stat(x) - stat(y) for x, y in pairs],
                    40,
                    agg_color,
                )


def boxplot_scores(
    ltype: list[TreeTypes],
    rtype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        xs = [node for node in _all_nodes(roots) if node.type in ltype]
        ys = [node for node in _all_nodes(roots) if node.type in rtype]
        title = f"""
# boxplots of left_min, left_max, left_avg, right_min, right_max, right_avg

tree_to_scores = {fn_txt(tree_to_scores)}

for every tree in source:
  for every node in tree:
    if node.type in {ltype}:
      left_min.append(min(tree_to_scores(node)))
      left_max.append(max(tree_to_scores(node)))
      left_avg.append(avg(tree_to_scores(node)))
    if node.type in {rtype}:
      right_min.append(min(tree_to_scores(node)))
      right_max.append(max(tree_to_scores(node)))
      right_avg.append(avg(tree_to_scores(node)))"""
        for _, getter, ax in _ax6(title, bool(xs) and bool(ys)):
            for i, (agg_name, agg_color, agg_fn) in enumerate(agg_list):

                def stat(t: Tree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]

                def label(sublabel: str):
                    return agg_name + (f"\n{sublabel}" if i == 1 else "")

                ax.boxplot(
                    [x, y],
                    labels=[label(str(ltype)), label(str(rtype))],
                    positions=[i, i + len(agg_list)],
                    boxprops=dict(color=agg_color),
                    capprops=dict(color=agg_color),
                    whiskerprops=dict(color=agg_color),
                    flierprops=dict(color=agg_color, markeredgecolor=agg_color),
                    medianprops=dict(color=agg_color),
                )


def histogram_top_bottom(
    ttype: list[TreeTypes],
    btype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    agg_tuple: AggTuple,
    score_getter: Tuple[str, ScoreGetter],
    detailed: bool,
):
    for roots, _source in input_data(detailed):
        agg_name, agg_color, agg_fn = agg_tuple
        score_name, getter = score_getter

        def stat(t: Tree) -> float:
            return _tree_to_float(t, tree_to_scores, getter, agg_fn)

        title = f"""
# histogram of top, bottom

tree_to_scores = {fn_txt(tree_to_scores)}
score = {score_name}

for every tree in source:
  for every node in tree:
    if node.type in {ttype}:
      top.append({agg_name}(tree_to_scores(node))))
    if node.type in {btype}:
      top.append({agg_name}(tree_to_scores(node))))
"""
        all_nodes = list(flatten((root.all_nodes() for root in roots)))

        def minmax(l: list[float]):
            return min(l), max(l)

        range = minmax([stat(tree) for tree in all_nodes])
        for type, ax in _ax2(title, ttype, btype):
            # ax.set_ylim((0, 3000))
            _hist(
                ax,
                [stat(tree) for tree in all_nodes if tree.type in type],
                30,
                agg_color,
                range=range,
            )
        y = [1 for tree in all_nodes if tree.type in ttype] + [
            0 for tree in all_nodes if tree.type in btype
        ]
        pred = [stat(tree) for tree in all_nodes if tree.type in ttype] + [
            stat(tree) for tree in all_nodes if tree.type in btype
        ]
        fpr, tpr, _thresholds = metrics.roc_curve(y, pred)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{score_name} {agg_name}"
        )

        fig, ax = plt.subplots()
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.75)
        fig.set_dpi(220)
        fig.set_figheight(7)
        display.plot(ax)
        plt.show()

def get_siblings_data(
        node_details: list[tuple[list[TreeTypes], list[TreeTypes],
                                 Fn[Tree, list[Score]]]],
        detailed: bool) -> dict[str, list[tuple[str, list[list[float]]]]]:
    result: dict[str, list[tuple[str, list[list[float]]]]] = {}
    for i, (roots, _) in enumerate(
            input_data(False)):  # set detailed to False, always one loop
        assert i <= 1
        # print(_score_getters)
        for score_name, score_getter in _score_getters:
            one_picture_data = []
            for agg_name, _, agg_fn in agg_list:
                single_panel_data = []
                for xtype, ytype, tree_to_scores in node_details:
                    single_type_pairs: list[float] = []
                    pairs = _pairs_siblings(roots, xtype, ytype)

                    def stat(t: Tree) -> float:
                        return _tree_to_float(t, tree_to_scores, score_getter,
                                           agg_fn)

                    for x, y in pairs:
                        single_type_pairs.append(stat(x) - stat(y))
                    single_panel_data.append(single_type_pairs)
                one_picture_data.append((agg_name, single_panel_data))
            result[score_name] = one_picture_data
    return result

def get_parent_child_data(
        node_details: list[tuple[list[TreeTypes], list[TreeTypes],
                                 Fn[Tree, list[Score]]]],
        detailed: bool) -> dict[str, list[tuple[str, list[list[float]]]]]:
    result: dict[str, list[tuple[str, list[list[float]]]]] = {}
    for i, (roots, _) in enumerate(
            input_data(False)):  # set detailed to False, always one loop
        assert i < 1
        # print(_score_getters)
        for score_name, score_getter in _score_getters:
            one_picture_data = []
            for agg_name, _, agg_fn in agg_list:
                single_panel_data = []
                for parenttype, childtype, tree_to_scores in node_details:
                    single_type_pairs: list[float] = []

                    pairs = _pairs_parent_child(roots, parenttype, childtype)
                    # TODO refactor this, it is practically the same as get_siblings_data

                    def stat(t: Tree) -> float:
                        return _tree_to_float(t, tree_to_scores, score_getter,
                                           agg_fn)

                    for x, y in pairs:
                        single_type_pairs.append(stat(x) - stat(y))
                    single_panel_data.append(single_type_pairs)
                one_picture_data.append((agg_name, single_panel_data))
            result[score_name] = one_picture_data
    return result