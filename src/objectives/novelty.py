from functools import total_ordering
from scipy.spatial import distance

KNN = 15
NOVELTY_ARCHIVE_MAX_SIZE = 50


class NoveltyArchive:
    def __init__(self, metric):
        self.novelty_metric = metric
        self.novel_items = []

    def size(self):
        return len(self.novel_items)

    def evaluate_novelty_score(self, item, n_items_list):
        # collect distances among archived novelty items
        distances = []
        for nov_item in self.novel_items:
            if nov_item.genome_id != item.genome_id:
                distances.append(self.novelty_metric(nov_item, item))
            else:
                print("Novelty Item is already in archive: %d" % nov_item.genome_id)

        # collect distances to the novelty items in the population
        for pop_item in n_items_list:
            if pop_item.genome_id != item.genome_id:
                distances.append(self.novelty_metric(pop_item, item))

        # calculate average KNN
        distances = sorted(distances)
        item.novelty = sum(distances[:KNN])/KNN

        # store novelty item
        self._add_novelty_item(item)

        return item.novelty

    def write_to_file(self, path):
        with open(path, "w") as f:
            for nov_item in self.novel_items:
                f.write(f"{nov_item}\n")

    def _add_novelty_item(self, item):
        item.in_archive = True

        if len(self.novel_items) >= NOVELTY_ARCHIVE_MAX_SIZE:
            if item > self.novel_items[-1]:
                self.novel_items[-1] = item
        else:
            self.novel_items.append(item)

        self.novel_items.sort(reverse=True)


@total_ordering
class NoveltyItem:
    def __init__(self, generation=-1, genome_id=-1, novelty=-1):
        self.generation = generation
        self.genome_id = genome_id
        self.novelty = novelty

        self.in_archive = False
        self.data = []

    def __str__(self):
        return f"NoveltyItem: id: {self.genome_id}, at generation: {self.generation},\
                              novelty: {self.novelty}\tdata: {self.data}"

    def _is_valid_operand(self, other):
        return hasattr(other, "novelty")

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented

        return self.novelty < other.novelty