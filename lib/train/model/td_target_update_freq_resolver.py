from lib.util.collection import first


class TDTargetUpdateFreqResolver:
    def __init__(self, schedule):
        self.schedule = schedule

    def current(self):
        return self.__current_schedule_item['freq']

    def __find_schedule_items_by(self, time):
        return list(filter(lambda it: it['from_time'] == time, self.schedule))

    def resolve(self, time):
        schedule_items = self.__find_schedule_items_by(time)

        if len(schedule_items) > 0:
            self.__current_schedule_item = first(schedule_items)

        return self.current()
