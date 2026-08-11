"""Broker integrations for client-backed simulation trading."""

from autostrategy.brokers.base import BrokerAdapter
from autostrategy.brokers.ft_client import FtClientBroker, FtClientError

__all__ = ["BrokerAdapter", "FtClientBroker", "FtClientError"]
