" Vim Editor Graphics
set number
set relativenumber
set tabstop=4
set shiftwidth=4
set autoindent
colorscheme slate

" Starts a VIM Server
if empty(v:servername) && exists('*remote_startserver')
	call remote_startserver('VIM')
endif

" Vimplug Automatic Download
let data_dir = has('nvim') ? stdpath('data') . '/site' : '~/.vim'
if empty(glob(data_dir . '/autoload/plug.vim'))
  silent execute '!curl -fLo '.data_dir.'/autoload/plug.vim --create-dirs  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
  autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif

"Vim Plugins
call plug#begin()
Plug 'lervag/vimtex' " VIMTeX plugin
Plug 'preservim/nerdtree' "Nerdtree plugin
Plug 'SirVer/ultisnips' "Ultisnips plugin
call plug#end()

" VimTeX Config
let g:vimtex_view_general_viewer = '/home/asingh/.local/bin/sumatrapdf.sh'
let g:vimtex_view_general_options = '-reuse-instance -forward-search @tex @line @pdf'

" NERDTree Config
autocmd VimEnter * NERDTree
let NERDTreeShowHidden=1

" Ultisnips Config
let g:UltiSnipsSnippetDirectories=[$HOME.'/.vim/UltiSnips']
let g:UltiSnipsExpandTrigger       = '<Tab>'    " use Tab to expand snippets
let g:UltiSnipsJumpForwardTrigger  = '<Tab>'    " use Tab to move forward through tabstops
let g:UltiSnipsJumpBackwardTrigger = '<S-Tab>'  " use Shift-Tab to move backward through tabstops
