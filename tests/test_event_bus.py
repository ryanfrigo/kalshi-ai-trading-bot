"""EventBus: subscribe, filter, unsubscribe, and a subscriber that fails. No network."""
import pytest

from src.events.event_bus import EventBus, EventData


@pytest.fixture
def bus():
    return EventBus()


def recorder():
    seen = []

    async def cb(event):
        seen.append(event)
    return cb, seen


def test_get_instance_is_shared_until_reset():
    EventBus.reset_instance()
    first = EventBus.get_instance()
    assert EventBus.get_instance() is first
    EventBus.reset_instance()
    assert EventBus.get_instance() is not first
    EventBus.reset_instance()


async def test_the_same_callback_and_filter_registers_once(bus):
    cb, seen = recorder()
    bus.subscribe("p", cb)
    bus.subscribe("p", cb)
    assert bus.subscriber_count("p") == 1
    await bus.publish("p", {})
    assert len(seen) == 1


async def test_the_same_callback_under_two_filters_registers_twice(bus):
    cb, _ = recorder()
    bus.subscribe("p", cb, ticker="A")
    bus.subscribe("p", cb, ticker="B")
    assert bus.subscriber_count("p") == 2


async def test_a_ticker_filter_matches_ticker_or_market_ticker_only(bus):
    cb, seen = recorder()
    bus.subscribe("p", cb, ticker="A")
    await bus.publish("p", {"ticker": "A"})
    await bus.publish("p", {"market_ticker": "A"})
    await bus.publish("p", {"ticker": "B"})
    await bus.publish("p", {})
    assert [e.data for e in seen] == [{"ticker": "A"}, {"market_ticker": "A"}]
    assert bus.stats == {"published": 4, "delivered": 2, "errors": 0}


async def test_unsubscribe_must_name_the_same_filter(bus):
    cb, _ = recorder()
    bus.subscribe("p", cb, ticker="A")
    bus.unsubscribe("p", cb)
    assert bus.subscriber_count("p") == 1
    bus.unsubscribe("p", cb, ticker="A")
    assert bus.subscriber_count("p") == 0


def test_unsubscribing_an_unknown_callback_changes_nothing(bus):
    cb, _ = recorder()
    other, _ = recorder()
    bus.subscribe("p", cb)
    bus.unsubscribe("p", other)
    bus.unsubscribe("never", other)
    assert bus.subscriber_count() == 1


def test_unsubscribe_all_for_one_type_leaves_the_others(bus):
    cb, _ = recorder()
    bus.subscribe("p", cb)
    bus.subscribe("q", cb)
    bus.unsubscribe_all("p")
    assert bus.subscriber_count("p") == 0 and bus.subscriber_count("q") == 1
    bus.unsubscribe_all()
    assert bus.subscriber_count() == 0


async def test_a_failing_subscriber_is_counted_and_the_next_one_still_runs(bus):
    async def boom(event):
        raise RuntimeError("subscriber failed")
    cb, seen = recorder()
    bus.subscribe("p", boom)
    bus.subscribe("p", cb)
    await bus.publish("p", {"x": 1})
    assert len(seen) == 1 and seen[0].event_type == "p"
    assert bus.stats == {"published": 1, "delivered": 1, "errors": 1}


async def test_a_subscriber_can_unsubscribe_during_publish_without_skipping_the_next(bus):
    cb, seen = recorder()

    async def once(event):
        bus.unsubscribe("p", once)
    bus.subscribe("p", once)
    bus.subscribe("p", cb)
    await bus.publish("p", {})
    assert len(seen) == 1 and bus.subscriber_count("p") == 1


async def test_publishing_with_no_subscribers_counts_only_published(bus):
    await bus.publish("p", {})
    assert bus.stats == {"published": 1, "delivered": 0, "errors": 0}


def test_stats_returns_a_copy(bus):
    bus.stats["published"] = 99
    assert bus.stats["published"] == 0


def test_an_event_without_a_ticker_matches_nothing():
    assert EventData("p", {}).matches_ticker("A") is False
