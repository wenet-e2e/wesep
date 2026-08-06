# -*- coding: utf-8 -*-

"""
Include YAML files within YAML
"""

import os.path
import re
from glob import iglob
from sys import version_info
from typing import Optional, Pattern, Sequence, Tuple

import yaml

from .readers import Reader, get_reader_class_by_name, get_reader_class_by_path

__all__ = ['YamlIncludeConstructor']

PYTHON_MAYOR_MINOR = '{0[0]}.{0[1]}'.format(version_info)

WILDCARDS_REGEX = re.compile(r'^.*(\*|\?|\[!?.+]).*$')


class YamlIncludeConstructor:
    """The `include constructor` for PyYAML Loaders

    Call :meth:`add_to_loader_class` or :meth:`yaml.Loader.add_constructor` to add it into loader.

    In YAML files, use ``!include`` to load other YAML files as below::

        !include [dir/**/*.yml, true]

    or::

        !include {pathname: dir/abc.yml, encoding: utf-8}

    """

    DEFAULT_ENCODING = 'utf-8'
    DEFAULT_TAG_NAME = '!include'

    def __init__(
            self,
            base_dir: str = '',
            encoding: str = '',
            reader_map: Optional[Sequence[Tuple[Pattern, Reader]]] = None
    ):
        """
        :param str base_dir: Base directory where search including YAML files

            :default: ``""``:  include YAML files from current working directory.

        :param str encoding: Encoding of the YAML files

            :default: ``""``:  Not specified

        :param Collection reader_map: A collection of `(path-pattern, reader-class)` tuple

            :default: ``None``: set :data:`readers.READER_TABLE` as default readers map
        """
        self._base_dir = base_dir
        self._encoding = encoding
        self._reader_map = reader_map

    def __call__(self, loader, node):
        args = []
        kwargs = {}
        if isinstance(node, yaml.nodes.ScalarNode):  # type: ignore
            args = [loader.construct_scalar(node)]
        elif isinstance(node, yaml.nodes.SequenceNode):  # type: ignore
            args = loader.construct_sequence(node)
        elif isinstance(node, yaml.nodes.MappingNode):  # type: ignore
            kwargs = loader.construct_mapping(node)
        else:
            raise TypeError('Un-supported YAML node {!r}'.format(node))
        return self.load(loader, *args, **kwargs)

    @property
    def base_dir(self) -> str:
        """Base directory where search including YAML files

        :rtype: str
        """
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: str):
        self._base_dir = value

    @property
    def encoding(self) -> str:
        """Encoding of the YAML files

        :rtype: str
        """
        return self._encoding

    @encoding.setter
    def encoding(self, value: str):
        self._encoding = value

    def load(
            self,
            loader,
            pathname: str,
            recursive: bool = False,
            encoding: str = '',
            reader: str = ''
    ):  # pylint:disable=too-many-arguments
        """Once add the constructor to PyYAML loader class,
        Loader will use this function to include other YAML files
        on parsing ``"!include"`` tag

        :param loader: Instance of PyYAML's loader class
        :param str pathname: pathname can be either absolute (like `/usr/src/Python-1.5/*.yml`)
            or relative (like `../../Tools/*/*.yml`), and can contain shell-style wildcards

        :param bool recursive: If recursive is true, the pattern ``"**"`` will match any files and zero
            or more directories and subdirectories.
            If the pattern is followed by an os.sep, only directories and subdirectories match.

            .. note:: Using the ``"**"`` pattern in large directory trees may consume an inordinate amount of time.

        :param str encoding: YAML file encoding

            :default: ``None``: Attribute :attr:`encoding` or constant :attr:`DEFAULT_ENCODING` will be used to open it

        :param str reader: name of the reader for loading files

            it's typically one of:

            - `ini`
            - `json`
            - `yaml`
            - `toml`
            - `txt`

            if not specified, reader would be decided by `reader_map` parameter passed in constructor

        :return: included YAML file, in Python data type

        .. warning:: It's called by :mod:`yaml`. Do NOT call it yourself.
        """
        if not encoding:
            encoding = self._encoding or self.DEFAULT_ENCODING
        if self._base_dir:
            pathname = os.path.join(self._base_dir, pathname)
        reader_clz = None
        if reader:
            reader_clz = get_reader_class_by_name(reader)
        if re.match(WILDCARDS_REGEX, pathname):
            result = []
            iterable = iglob(pathname, recursive=recursive)
            for path in filter(os.path.isfile, iterable):
                if reader_clz:
                    result.append(reader_clz(path, encoding=encoding, loader_class=type(loader))())
                else:
                    result.append(self._read_file(path, loader, encoding))
            return result
        if reader_clz:
            return reader_clz(pathname, encoding=encoding, loader_class=type(loader))()
        return self._read_file(pathname, loader, encoding)

    def _read_file(self, path, loader, encoding):
        reader_clz = get_reader_class_by_path(path, self._reader_map)
        reader_obj = reader_clz(path, encoding=encoding, loader_class=type(loader))
        return reader_obj()

    @classmethod
    def add_to_loader_class(
            cls,
            loader_class=None,
            tag: str = None,
            **kwargs
    ):
        """
        Create an instance of the constructor, and add it to the YAML `Loader` class

        :param loader_class: The `Loader` **class** add constructor to.

            .. attention:: This parameter **SHOULD** be a **class type**, **NOT** an object.

            It's one of followings:

                - :class:`yaml.BaseLoader`
                - :class:`yaml.UnSafeLoader`
                - :class:`yaml.SafeLoader`
                - :class:`yaml.Loader`
                - :class:`yaml.FullLoader`
                - :class:`yaml.CBaseLoader`
                - :class:`yaml.CUnSafeLoader`
                - :class:`yaml.CSafeLoader`
                - :class:`yaml.CLoader`
                - :class:`yaml.CFullLoader`

            :default: ``None``:

                - When :mod:`pyyaml` `3.*`: it's :class:`yaml.Loader`
                - When :mod:`pyyaml` `5.*`: it's :class:`yaml.FullLoader`

        :param str tag: Tag's name of the include constructor.

          :default: ``""``: Use :attr:`DEFAULT_TAG_NAME` as tag name.

        :param kwargs: Arguments passed to construct function

        :return: New created object
        :rtype: YamlIncludeConstructor
        """
        if tag is None:
            tag = ''
        tag = tag.strip()
        if not tag:
            tag = cls.DEFAULT_TAG_NAME
        if not tag.startswith('!'):
            raise ValueError('`tag` argument should start with character "!"')
        instance = cls(**kwargs)
        yaml.add_constructor(tag, instance, loader_class)
        return instance
