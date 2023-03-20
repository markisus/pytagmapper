class MinHeap:
    def __init__(self):
        self.priorities = []
        self.id_to_idx = {}

    def _bubble_up(self, idx):
        parent_idx = int((idx-1)/2)
        task_id, priority = self.priorities[idx]
        parent_task_id, parent_priority = self.priorities[parent_idx]
        bubbled = False
        while parent_priority > priority and parent_task_id != task_id:
            self.priorities[parent_idx] = (task_id, priority)
            self.priorities[idx] = (parent_task_id, parent_priority)
            self.id_to_idx[parent_task_id] = idx
            self.id_to_idx[task_id] = parent_idx
            bubbled = True
            # advance the parent
            idx = parent_idx
            parent_idx = int((idx-1)/2)
            parent_task_id, parent_priority = self.priorities[parent_idx]
        return bubbled


    def _bubble_down(self, idx):
        child1_idx = int(2*idx+1)
        while child1_idx < len(self.priorities):
            min_child_idx = child1_idx
            min_child_id, min_child_priority = self.priorities[child1_idx]

            child2_idx = child1_idx+1
            if child2_idx < len(self.priorities):
                child2_idx = int(2*idx+2)
                child2_id, child2_priority = self.priorities[child2_idx]
                if child2_priority  < min_child_priority:
                    min_child_idx = child2_idx
                    min_child_id = child2_id
                    min_child_priority = child2_priority

            task_id, priority = self.priorities[idx]
            if min_child_priority < priority:
                self.priorities[min_child_idx] = (task_id, priority)
                self.priorities[idx] = (min_child_id, min_child_priority)
                self.id_to_idx[min_child_id] = idx
                self.id_to_idx[task_id] = min_child_idx
                idx = min_child_idx
                child1_idx = int(2*idx+1)
            else:
                return

    def upsert(self, task_id, priority):
        idx = self.id_to_idx.get(task_id, None)
        if idx is None:
            self.priorities.append((task_id, priority))
            self.id_to_idx[task_id] = len(self.priorities) - 1
            self._bubble_up(len(self.priorities)-1)
            return

        self.priorities[idx] = (task_id, priority)
        if not self._bubble_up(idx):
            self._bubble_down(idx)

    def pop(self):
        result = self.priorities[0]
        self.priorities[0] = self.priorities[-1]
        self.priorities.pop()

        # fix ids
        del self.id_to_idx[result[0]]

        if len(self.priorities):
            self.id_to_idx[self.priorities[0][0]] = 0
            self._bubble_down(0)

        return result

    def __len__(self):
        return len(self.priorities)

if __name__ == "__main__":
    import random

    ps = [95, 43, 66, 29, 21, 14, 56, 33, 88]
    n = 8

    h = MinHeap()
    for i in range(n):
        h.upsert(i, ps[i])
    h.upsert(3, 1000)
    h.upsert(3, 1)

    for i in range(n):
        print(h.pop())
